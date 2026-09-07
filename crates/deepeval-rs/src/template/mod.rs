//! Jinja-compatible prompt templating.
//!
//! This module resolves and renders metric prompt templates using
//! [`minijinja`]. Templates are embedded at compile time via `include_str!`
//! and looked up by `(metric_class_name, method)`.

use std::collections::HashMap;

use minijinja::{Environment, Value};

use crate::error::TemplateError;

/// A registry of embedded prompt templates.
///
/// Templates are keyed by `(class_name, method_name)` and rendered with
/// [`minijinja`]. This mirrors deepeval's `resolve_template` over its bundled
/// `templates.json`.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    templates: HashMap<(String, String), String>,
}

impl TemplateRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template under `(class_name, method_name)`.
    pub fn register(
        &mut self,
        class_name: impl Into<String>,
        method_name: impl Into<String>,
        template: impl Into<String>,
    ) {
        self.templates
            .insert((class_name.into(), method_name.into()), template.into());
    }

    /// Resolve and render a template with the given context.
    ///
    /// Returns [`TemplateError::NotFound`] if the template is not registered,
    /// or [`TemplateError::Render`] if rendering fails.
    pub fn resolve(
        &self,
        class_name: &str,
        method_name: &str,
        context: &HashMap<String, Value>,
    ) -> Result<String, TemplateError> {
        let template = self
            .templates
            .get(&(class_name.to_string(), method_name.to_string()))
            .ok_or_else(|| TemplateError::NotFound {
                class: class_name.to_string(),
                method: method_name.to_string(),
            })?;

        let mut env = Environment::new();
        env.add_template("t", template)
            .map_err(|e| TemplateError::Render(method_name.to_string(), e.to_string()))?;
        let tmpl = env
            .get_template("t")
            .map_err(|e| TemplateError::Render(method_name.to_string(), e.to_string()))?;
        tmpl.render(context)
            .map_err(|e| TemplateError::Render(method_name.to_string(), e.to_string()))
    }
}

/// Resolve and render a template from a registry.
///
/// Convenience wrapper over [`TemplateRegistry::resolve`].
pub fn resolve_template(
    registry: &TemplateRegistry,
    class_name: &str,
    method_name: &str,
    context: &HashMap<String, Value>,
) -> Result<String, TemplateError> {
    registry.resolve(class_name, method_name, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::Value;

    fn context(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::from(*v)))
            .collect()
    }

    #[test]
    fn renders_simple_template() {
        let mut registry = TemplateRegistry::new();
        registry.register("MyMetric", "greet", "Hello, {{ name }}!");

        let ctx = context(&[("name", "world")]);
        let out = registry.resolve("MyMetric", "greet", &ctx).unwrap();
        assert_eq!(out, "Hello, world!");
    }

    #[test]
    fn renders_conditionals() {
        let mut registry = TemplateRegistry::new();
        registry.register(
            "MyMetric",
            "check",
            "{% if strict %}strict{% else %}loose{% endif %}",
        );

        let ctx = context(&[("strict", "true")]);
        let out = registry.resolve("MyMetric", "check", &ctx).unwrap();
        assert_eq!(out, "strict");
    }

    #[test]
    fn missing_template_returns_not_found() {
        let registry = TemplateRegistry::new();
        let err = registry
            .resolve("Missing", "method", &HashMap::new())
            .unwrap_err();
        assert!(matches!(err, TemplateError::NotFound { .. }));
    }

    #[test]
    fn resolve_template_wrapper() {
        let mut registry = TemplateRegistry::new();
        registry.register("MyMetric", "greet", "Hi {{ name }}");
        let ctx = context(&[("name", "bob")]);
        let out = resolve_template(&registry, "MyMetric", "greet", &ctx).unwrap();
        assert_eq!(out, "Hi bob");
    }
}
