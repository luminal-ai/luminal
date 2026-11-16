//! The byte form of a code object: JSON text, read version first.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde::de::DeserializeOwned;

/// Read a code object saved under `format` by the `backend` backend.
///
/// The version is read alone first: a program saved under another version
/// is refused by number, and a field that changed with the format is never
/// what fails.
pub fn read<T: DeserializeOwned>(bytes: &[u8], format: u32, backend: &str) -> Result<T> {
    #[derive(Deserialize)]
    struct Version {
        format: u32,
    }
    let version: Version = serde_json::from_slice(bytes)
        .with_context(|| format!("reading a {backend} code object's format"))?;
    if version.format != format {
        bail!(
            "this program was saved in {backend} code format {}; this build reads format \
             {format}",
            version.format
        );
    }
    serde_json::from_slice(bytes).with_context(|| format!("reading a {backend} code object"))
}

/// Bytes as a hex string: a constant's elements stay readable in the saved
/// form without a JSON number per byte. For `#[serde(with = "hex")]` on a
/// `Vec<u8>`; [`hex::option`] is the same for an `Option<Vec<u8>>`.
pub mod hex {
    use serde::{Deserialize, Deserializer, Serializer};

    fn encode(bytes: &[u8]) -> String {
        let mut text = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            text.push_str(&format!("{byte:02x}"));
        }
        text
    }

    fn decode<E: serde::de::Error>(text: &str) -> Result<Vec<u8>, E> {
        if !text.is_ascii() || !text.len().is_multiple_of(2) {
            return Err(E::custom(
                "a constant's hex string is not an even number of hex digits",
            ));
        }
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).map_err(E::custom))
            .collect()
    }

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        decode(&String::deserialize(deserializer)?)
    }

    pub mod option {
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S: Serializer>(
            bytes: &Option<Vec<u8>>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            match bytes {
                Some(bytes) => serializer.serialize_some(&super::encode(bytes)),
                None => serializer.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Option<Vec<u8>>, D::Error> {
            Option::<String>::deserialize(deserializer)?
                .map(|text| super::decode(&text))
                .transpose()
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Saved {
        format: u32,
        #[serde(with = "hex")]
        bytes: Vec<u8>,
        #[serde(with = "hex::option", default, skip_serializing_if = "Option::is_none")]
        maybe: Option<Vec<u8>>,
    }

    #[test]
    fn bytes_are_spelled_in_hex_and_read_back() {
        let saved = Saved {
            format: 3,
            bytes: vec![0, 1, 254, 255],
            maybe: Some(vec![16]),
        };
        let text = serde_json::to_string(&saved).unwrap();
        assert_eq!(text, r#"{"format":3,"bytes":"0001feff","maybe":"10"}"#);
        assert_eq!(read::<Saved>(text.as_bytes(), 3, "test").unwrap(), saved);

        let absent: Saved = read(br#"{"format":3,"bytes":""}"#, 3, "test").unwrap();
        assert_eq!(absent.maybe, None);
        assert!(read::<Saved>(br#"{"format":3,"bytes":"0g"}"#, 3, "test").is_err());
        assert!(read::<Saved>(br#"{"format":3,"bytes":"000"}"#, 3, "test").is_err());
    }

    /// The version decides before any other field is read, so a field that
    /// changed with the format is not what fails.
    #[test]
    fn another_format_is_refused_by_number() {
        let error = read::<Saved>(br#"{"format":4,"bytes":7}"#, 3, "test").unwrap_err();
        assert!(
            error.to_string().contains("test code format 4"),
            "{error:#}"
        );
        assert!(error.to_string().contains("reads format 3"), "{error:#}");
    }
}
