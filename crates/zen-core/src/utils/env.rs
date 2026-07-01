//! Environment variable utilities with backward compatibility support.
//!
//! This module provides helpers for environment variable migration from
//! `zenllm_*` to `ZENTRIX_*` naming while maintaining backward compatibility.

use tracing::warn;

/// Read an environment variable with fallback to legacy name.
///
/// Checks the new name first, then falls back to the old name if not found.
/// Logs a deprecation warning when the old name is used.
///
/// # Example
/// ```no_run
/// use zen_core::utils::env::env_or_legacy;
///
/// if let Some(value) = env_or_legacy("ZENTRIX_DEBUG", "zenllm_DEBUG") {
///     println!("Debug mode: {}", value);
/// }
/// ```
pub fn env_or_legacy(new_name: &str, old_name: &str) -> Option<String> {
    std::env::var(new_name).ok().or_else(|| {
        std::env::var(old_name).ok().map(|val| {
            warn!(
                "Environment variable {old_name} is deprecated, please use {new_name} instead"
            );
            val
        })
    })
}

/// Check if an environment variable is set to "1" with legacy fallback.
///
/// Returns `true` if either the new or old variable is set to "1".
/// Logs a deprecation warning when the old name is used.
///
/// # Example
/// ```no_run
/// use zen_core::utils::env::is_flag_enabled;
///
/// if is_flag_enabled("ZENTRIX_NO_MMAP", "ZEN_NO_MMAP") {
///     println!("Memory mapping disabled");
/// }
/// ```
pub fn is_flag_enabled(new_name: &str, old_name: &str) -> bool {
    env_or_legacy(new_name, old_name).is_some_and(|x| x == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_or_legacy_prefers_new() {
        std::env::set_var("TEST_NEW", "new_value");
        std::env::set_var("TEST_OLD", "old_value");
        
        let result = env_or_legacy("TEST_NEW", "TEST_OLD");
        assert_eq!(result, Some("new_value".to_string()));
        
        std::env::remove_var("TEST_NEW");
        std::env::remove_var("TEST_OLD");
    }

    #[test]
    fn test_env_or_legacy_falls_back() {
        std::env::set_var("TEST_OLD2", "old_value");
        
        let result = env_or_legacy("TEST_NEW2", "TEST_OLD2");
        assert_eq!(result, Some("old_value".to_string()));
        
        std::env::remove_var("TEST_OLD2");
    }

    #[test]
    fn test_is_flag_enabled() {
        std::env::set_var("TEST_FLAG_NEW", "1");
        assert!(is_flag_enabled("TEST_FLAG_NEW", "TEST_FLAG_OLD"));
        
        std::env::remove_var("TEST_FLAG_NEW");
        std::env::set_var("TEST_FLAG_OLD", "1");
        assert!(is_flag_enabled("TEST_FLAG_NEW", "TEST_FLAG_OLD"));
        
        std::env::remove_var("TEST_FLAG_OLD");
        assert!(!is_flag_enabled("TEST_FLAG_NEW", "TEST_FLAG_OLD"));
    }
}
