//! Useful macros for the naturalneighbor3d library.
//!
//! This module provides macros that help with error handling, validation,
//! and common patterns used throughout the library.

/// Macro for creating interpolation errors with context.
#[macro_export]
macro_rules! interpolation_error {
    ($variant:ident { $($field:ident: $value:expr),* $(,)? }) => {
        $crate::errors::InterpolationError::$variant {
            $($field: $value),*
        }
    };
    ($variant:ident($value:expr)) => {
        $crate::errors::InterpolationError::$variant($value)
    };
}

/// Macro for early return with interpolation error.
#[macro_export]
macro_rules! bail {
    ($msg:expr) => {
        return Err($crate::errors::InterpolationError::InvalidInput {
            message: $msg.to_string()
        });
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::errors::InterpolationError::InvalidInput {
            message: format!($fmt, $($arg)*)
        });
    };
}

/// Macro for ensuring conditions with error on failure.
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !($cond) {
            $crate::bail!($msg);
        }
    };
    ($cond:expr, $fmt:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::bail!($fmt, $($arg)*);
        }
    };
}

/// Macro for validating array shapes.
#[macro_export]
macro_rules! validate_shape {
    ($array:expr, $expected:expr, $name:expr) => {
        let shape = $array.shape();
        if shape != $expected {
            return Err($crate::interpolation_error!(InvalidPointsShape {
                shape: shape.to_vec()
            }));
        }
    };
}

/// Macro for safe array indexing with bounds checking in debug mode.
#[macro_export]
macro_rules! safe_get {
    ($array:expr, $index:expr) => {{
        debug_assert!(
            $index < $array.len(),
            "Index {} out of bounds for array of length {}",
            $index,
            $array.len()
        );
        unsafe { $array.get_unchecked($index) }
    }};
}

/// Macro for safe mutable array indexing with bounds checking in debug mode.
#[macro_export]
macro_rules! safe_get_mut {
    ($array:expr, $index:expr) => {{
        debug_assert!(
            $index < $array.len(),
            "Index {} out of bounds for array of length {}",
            $index,
            $array.len()
        );
        unsafe { $array.get_unchecked_mut($index) }
    }};
}

#[cfg(test)]
mod tests {
    use crate::errors::InterpolationError;

    #[test]
    fn test_interpolation_error_macro() {
        let error = interpolation_error!(InvalidInput {
            message: "test error".to_string()
        });

        match error {
            InterpolationError::InvalidInput { message } => {
                assert_eq!(message, "test error");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_ensure_macro() {
        let result = (|| -> Result<(), InterpolationError> {
            ensure!(true, "Should not fail");
            Ok(())
        })();
        assert!(result.is_ok());

        let result = (|| -> Result<(), InterpolationError> {
            ensure!(false, "Should fail");
            Ok(())
        })();
        assert!(result.is_err());
    }

    #[test]
    fn test_bail_macro() {
        let result = (|| -> Result<(), InterpolationError> {
            bail!("Test error");
        })();

        assert!(result.is_err());
        if let Err(InterpolationError::InvalidInput { message }) = result {
            assert_eq!(message, "Test error");
        } else {
            panic!("Wrong error type");
        }
    }
}
