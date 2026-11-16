use serde::{Deserialize, Serialize};

/// What the value in a slot is.
///
/// The kinds mirror what torch can pass a compiled graph: a tensor, the
/// Python scalar a symbolic int, float or bool placeholder arrives as, or
/// anything else. A kind says what the value *is*, never what a backend does
/// with it: a tensor no node reads is still a `Tensor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotKind {
    Tensor,
    Int,
    Float,
    Bool,
    /// `None`, a string, or an opaque object.
    Object,
}

impl SlotKind {
    pub fn name(self) -> &'static str {
        match self {
            SlotKind::Tensor => "tensor",
            SlotKind::Int => "int",
            SlotKind::Float => "float",
            SlotKind::Bool => "bool",
            SlotKind::Object => "object",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SlotKind;

    const ALL: [SlotKind; 5] = [
        SlotKind::Tensor,
        SlotKind::Int,
        SlotKind::Float,
        SlotKind::Bool,
        SlotKind::Object,
    ];

    /// A saved program stores one kind per slot. The stored spelling is the
    /// kind's name, and it reads back as the same kind.
    #[test]
    fn a_list_of_kinds_round_trips_as_their_names() {
        let written = serde_json::to_string(&ALL.to_vec()).unwrap();
        assert_eq!(written, r#"["tensor","int","float","bool","object"]"#);
        let read: Vec<SlotKind> = serde_json::from_str(&written).unwrap();
        assert_eq!(read, ALL);
        for kind in ALL {
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{}\"", kind.name())
            );
        }
    }

    /// A kind this build does not know is refused by name, not read as
    /// something else.
    #[test]
    fn an_unknown_kind_is_refused_by_name() {
        let error = serde_json::from_str::<SlotKind>("\"complex\"").unwrap_err();
        assert!(error.to_string().contains("complex"), "{error}");
    }
}
