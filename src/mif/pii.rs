//! PII detection and redaction for MIF exports.

use crate::mif::schema::MifRedaction;

/// Compiled PII patterns for detection.
pub struct PiiPatterns {
    email: regex::Regex,
    phone: regex::Regex,
    ssn: regex::Regex,
    api_key: regex::Regex,
    credit_card: regex::Regex,
    iban: regex::Regex,
    intl_phone: regex::Regex,
}

impl Default for PiiPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl PiiPatterns {
    pub fn new() -> Self {
        Self {
            email: regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
            phone: regex::Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap(),
            ssn: regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            api_key: regex::Regex::new(
                r#"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['"]?[\w-]{16,}['"]?"#,
            )
            .unwrap(),
            credit_card: regex::Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap(),
            // IBAN: 2-letter country code + 2 check digits + up to 30 alphanumeric
            iban: regex::Regex::new(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b").unwrap(),
            // International phone: +country code then 7-14 digits with optional separators
            intl_phone: regex::Regex::new(r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}").unwrap(),
        }
    }

    /// Detect and redact PII from content.
    ///
    /// Returns `(redacted_content, redaction_records, pii_found)`.
    pub fn redact(&self, content: &str) -> (String, Vec<MifRedaction>, bool) {
        let mut redacted = content.to_string();
        let mut redactions = Vec::new();
        let mut pii_found = false;

        let patterns: &[(&regex::Regex, &str, &str)] = &[
            (&self.email, "email", "[REDACTED:email]"),
            (&self.intl_phone, "phone", "[REDACTED:phone]"),   // international first (more specific)
            (&self.phone, "phone", "[REDACTED:phone]"),         // then US format
            (&self.ssn, "ssn", "[REDACTED:ssn]"),
            (&self.iban, "iban", "[REDACTED:iban]"),
            (&self.api_key, "api_key", "[REDACTED:api_key]"),
            (&self.credit_card, "credit_card", "[REDACTED:credit_card]"),
        ];

        for (regex, kind, replacement) in patterns {
            for m in regex.find_iter(content) {
                pii_found = true;
                redactions.push(MifRedaction {
                    redaction_type: kind.to_string(),
                    original_length: m.as_str().len(),
                    position: (m.start(), m.end()),
                });
            }
            redacted = regex.replace_all(&redacted, *replacement).to_string();
        }

        (redacted, redactions, pii_found)
    }

    /// Check if content contains any secrets (API keys, tokens, passwords).
    pub fn has_secrets(&self, content: &str) -> bool {
        self.api_key.is_match(content)
    }
}
