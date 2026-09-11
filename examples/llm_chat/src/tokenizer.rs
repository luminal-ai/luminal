use crate::checkpoint::read_json;
use anyhow::{Context, Result, anyhow, ensure};
use minijinja::{Environment, Error, ErrorKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}
impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub struct ChatTokenizer {
    pub tokenizer: Tokenizer,
    pub stop_tokens: BTreeSet<u32>,
    template: String,
    config: Value,
}
impl ChatTokenizer {
    pub fn load(directory: &Path, model_config: &Value) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let config = read_json(&directory.join("tokenizer_config.json"))?;
        let template_path = directory.join("chat_template.jinja");
        let template = if template_path.exists() {
            std::fs::read_to_string(template_path)?
        } else {
            match &config["chat_template"] {
                Value::String(s) => s.clone(),
                Value::Array(templates) => templates
                    .iter()
                    .find(|t| t["name"] == "default")
                    .and_then(|t| t["template"].as_str())
                    .ok_or_else(|| anyhow!("tokenizer_config.json has no default chat template"))?
                    .into(),
                Value::Object(templates) => templates
                    .get("default")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("tokenizer_config.json has no default chat template"))?
                    .into(),
                _ => {
                    return Err(anyhow!(
                        "checkpoint needs chat_template.jinja or tokenizer_config.json chat_template"
                    ));
                }
            }
        };
        let mut stop_tokens = BTreeSet::new();
        let mut add_ids = |v: &Value| -> Result<()> {
            let ids: Vec<_> = match v {
                Value::Null => vec![],
                Value::Array(v) => v.iter().collect(),
                _ => vec![v],
            };
            for id in ids {
                stop_tokens.insert(u32::try_from(
                    id.as_u64().ok_or_else(|| anyhow!("invalid eos_token_id"))?,
                )?);
            }
            Ok(())
        };
        add_ids(&model_config["eos_token_id"])?;
        add_ids(&model_config["text_config"]["eos_token_id"])?;
        let generation = directory.join("generation_config.json");
        if generation.exists() {
            add_ids(&read_json(&generation)?["eos_token_id"])?;
        }
        if let Some(eos) = special_token(&config["eos_token"])
            && let Some(id) = tokenizer.token_to_id(eos)
        {
            stop_tokens.insert(id);
        }
        ensure!(
            !stop_tokens.is_empty(),
            "checkpoint declares no usable EOS token"
        );
        Ok(Self {
            tokenizer,
            stop_tokens,
            template,
            config,
        })
    }
    pub fn encode_chat(&self, messages: &[Message], enable_thinking: bool) -> Result<Vec<u32>> {
        let rendered = render(&self.template, &self.config, messages, enable_thinking)?;
        // The chat template already inserts BOS/EOS and role markers.
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| anyhow!("tokenize chat: {e}"))?;
        ensure!(
            !encoding.is_empty(),
            "chat template rendered an empty prompt"
        );
        Ok(encoding.get_ids().to_vec())
    }
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("decode tokens: {e}"))
    }
}
fn special_token(v: &Value) -> Option<&str> {
    v.as_str().or_else(|| v["content"].as_str())
}
fn render(
    template: &str,
    config: &Value,
    messages: &[Message],
    enable_thinking: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    let mut context = config.as_object().cloned().unwrap_or_default();
    for (name, value) in context.iter_mut() {
        if name.ends_with("_token")
            && let Some(token) = special_token(value)
        {
            *value = Value::String(token.into());
        }
    }
    context.insert("messages".into(), serde_json::to_value(messages)?);
    context.insert("add_generation_prompt".into(), true.into());
    context.insert("enable_thinking".into(), enable_thinking.into());
    env.template_from_str(template)?
        .render(context)
        .context("render checkpoint chat template")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_template_controls_roles_and_special_tokens() {
        let config = serde_json::json!({"bos_token":{"content":"<bos>"}});
        let template = "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content.strip() }};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}{% if enable_thinking %}<think>{% endif %}";
        assert_eq!(
            render(template, &config, &[Message::new("user", " hello ")], false).unwrap(),
            "<bos>user:hello;assistant:"
        );
        assert!(render("{{ raise_exception('bad roles') }}", &config, &[], false).is_err());
    }
}
