//! The `cargo release` tool.
//!
//! Bumps the workspace version, moves the `[Unreleased]` CHANGELOG entries into
//! a new version section, refreshes `Cargo.lock`, commits, tags `vX.Y.Z`, and
//! pushes the tag — which triggers the GitHub Actions release workflow.
//!
//! Usage (via the `release` cargo alias):
//!
//! ```text
//! cargo release major          # 0.1.0 -> 1.0.0
//! cargo release minor          # 0.1.0 -> 0.2.0
//! cargo release patch          # 0.1.0 -> 0.1.1
//! cargo release 1.2.3          # explicit version
//! cargo release patch --dry-run  # preview without committing or pushing
//! ```

use std::process::Command;

use clap::Parser;

/// Bump the version, tag, and push a release.
#[derive(Debug, Parser)]
#[command(name = "release", about = "Bump the version, tag, and push a release")]
struct Cli {
    /// `major`, `minor`, `patch`, or an explicit semver like `1.2.3`.
    bump: String,

    /// Print what would happen without committing or pushing.
    #[arg(long, global = true)]
    dry_run: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum Bump {
    Major,
    Minor,
    Patch,
    Exact(String),
}

fn parse_bump(s: &str) -> Result<Bump, String> {
    match s {
        "major" => Ok(Bump::Major),
        "minor" => Ok(Bump::Minor),
        "patch" => Ok(Bump::Patch),
        _ => {
            let parts: Vec<&str> = s.split('.').collect();
            if parts.len() == 3 && parts.iter().all(|p| p.parse::<u64>().is_ok()) {
                Ok(Bump::Exact(s.to_string()))
            } else {
                Err(format!(
                    "invalid bump `{s}`: expected `major`, `minor`, `patch`, or `X.Y.Z`"
                ))
            }
        }
    }
}

fn next_version(current: &str, bump: &Bump) -> Result<String, String> {
    let parts: Vec<u64> = current
        .split('.')
        .map(|p| p.parse::<u64>().map_err(|_| format!("invalid current version `{current}`")))
        .collect::<Result<_, _>>()?;
    if parts.len() != 3 {
        return Err(format!("invalid current version `{current}`"));
    }
    let (major, minor, patch) = (parts[0], parts[1], parts[2]);
    Ok(match bump {
        Bump::Major => format!("{}.0.0", major + 1),
        Bump::Minor => format!("{major}.{}.0", minor + 1),
        Bump::Patch => format!("{major}.{minor}.{}", patch + 1),
        Bump::Exact(v) => v.clone(),
    })
}

/// Read the `[workspace.package] version` from the root `Cargo.toml`.
fn read_workspace_version(cargo_toml: &str) -> Result<String, String> {
    let mut in_workspace_package = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_workspace_package = trimmed == "[workspace.package]";
            continue;
        }
        if in_workspace_package && trimmed.starts_with("version = ") {
            let v = trimmed
                .trim_start_matches("version = ")
                .trim()
                .trim_matches('"');
            return Ok(v.to_string());
        }
    }
    Err("could not find `[workspace.package] version` in Cargo.toml".into())
}

/// Rewrite the `[workspace.package] version` in the root `Cargo.toml`.
fn set_workspace_version(cargo_toml: &str, new_version: &str) -> String {
    let mut in_workspace_package = false;
    let mut out = String::new();
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_workspace_package = trimmed == "[workspace.package]";
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if in_workspace_package && trimmed.starts_with("version = ") {
            let indent = &line[..line.len() - line.trim_start().len()];
            out.push_str(&format!("{indent}version = \"{new_version}\""));
            out.push('\n');
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Set the `version` field of a path dependency (e.g. `deepeval-rs`) in a
/// crate manifest, so the published dependency resolves on crates.io.
fn set_dep_version(manifest: &str, dep: &str, new_version: &str) -> String {
    let mut out = String::new();
    for line in manifest.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&format!("{dep} = {{")) {
            let open = line.find('{').unwrap_or(0);
            let close = line.rfind('}').unwrap_or(line.len());
            let inner = &line[open + 1..close];
            let leading = inner.len() - inner.trim_start().len();
            let trailing = inner.len() - inner.trim_end().len();
            let mut fields: Vec<String> = Vec::new();
            for part in inner.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                if part.starts_with("version = ") {
                    fields.push(format!("version = \"{new_version}\""));
                } else {
                    fields.push(part.to_string());
                }
            }
            if !fields.iter().any(|f| f.starts_with("version = ")) {
                fields.push(format!("version = \"{new_version}\""));
            }
            let prefix = &line[..open];
            let suffix = &line[close + 1..];
            let pad_lead = " ".repeat(leading);
            let pad_trail = " ".repeat(trailing);
            out.push_str(&format!(
                "{prefix}{{{pad_lead}{}{pad_trail}}}{suffix}",
                fields.join(", ")
            ));
            out.push('\n');
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Move the `[Unreleased]` body into a new `[X.Y.Z] - <date>` section.
fn update_changelog(changelog: &str, new_version: &str, date: &str) -> Result<String, String> {
    let marker = "## [Unreleased]";
    let start = changelog
        .find(marker)
        .ok_or_else(|| "no `[Unreleased]` section in CHANGELOG.md".to_string())?;
    let after = &changelog[start + marker.len()..];
    let next = after
        .find("\n## [")
        .ok_or_else(|| "no version section after `[Unreleased]` in CHANGELOG.md".to_string())?;
    let body_start = start + marker.len();
    let body_end = body_start + next;
    let body = changelog[body_start..body_end].trim();

    let mut out = String::new();
    out.push_str(&changelog[..start + marker.len()]);
    out.push_str("\n\n");
    out.push_str(&format!("## [{new_version}] - {date}\n\n"));
    out.push_str(body);
    out.push_str("\n\n");
    out.push_str(&changelog[body_end..]);
    Ok(out)
}

fn git(args: &[&str]) -> Result<(), String> {
    let status = Command::new("git")
        .args(args)
        .status()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !status.success() {
        return Err(format!("`git {}` failed", args.join(" ")));
    }
    Ok(())
}

fn git_output(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err(format!("`git {}` failed", args.join(" ")));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn today() -> String {
    let out = Command::new("date")
        .args(["+%Y-%m-%d"])
        .output()
        .expect("failed to run date");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let bump = parse_bump(&cli.bump)?;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let cargo_toml_path = root.join("Cargo.toml");
    let changelog_path = root.join("CHANGELOG.md");
    let cli_manifest_path = root.join("crates/deepeval/Cargo.toml");

    let cargo_toml = std::fs::read_to_string(&cargo_toml_path)
        .map_err(|e| format!("failed to read Cargo.toml: {e}"))?;
    let current = read_workspace_version(&cargo_toml)?;
    let new = next_version(&current, &bump)?;

    if new == current {
        return Err(format!("version is already {current}; nothing to bump"));
    }

    // Refuse to run on a dirty tree so we never commit unrelated changes.
    let dirty = git_output(&["status", "--porcelain"])?;
    if !dirty.is_empty() {
        return Err("working tree is not clean; commit or stash changes first".into());
    }

    let tag = format!("v{new}");
    let branch = git_output(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let date = today();

    println!("Current version: {current}");
    println!("New version:      {new}");
    println!("Tag:              {tag}");
    println!("Branch:           {branch}");
    println!("Date:             {date}");

    if cli.dry_run {
        println!("\n[dry-run] would update Cargo.toml, crates/deepeval/Cargo.toml, CHANGELOG.md, and Cargo.lock;");
        println!("[dry-run] commit `chore: release {tag}`, tag `{tag}`, and push `{branch}` + `{tag}`.");
        return Ok(());
    }

    // 1. Bump the version in the root Cargo.toml.
    let new_cargo_toml = set_workspace_version(&cargo_toml, &new);
    std::fs::write(&cargo_toml_path, new_cargo_toml)
        .map_err(|e| format!("failed to write Cargo.toml: {e}"))?;

    // 1b. Keep the CLI's path dependency on the library in sync so it resolves
    // on crates.io.
    let cli_manifest = std::fs::read_to_string(&cli_manifest_path)
        .map_err(|e| format!("failed to read crates/deepeval/Cargo.toml: {e}"))?;
    let new_cli_manifest = set_dep_version(&cli_manifest, "deepeval-rs", &new);
    std::fs::write(&cli_manifest_path, new_cli_manifest)
        .map_err(|e| format!("failed to write crates/deepeval/Cargo.toml: {e}"))?;

    // 2. Move Unreleased entries into the new version section.
    let changelog = std::fs::read_to_string(&changelog_path)
        .map_err(|e| format!("failed to read CHANGELOG.md: {e}"))?;
    let new_changelog = update_changelog(&changelog, &new, &date)?;
    std::fs::write(&changelog_path, new_changelog)
        .map_err(|e| format!("failed to write CHANGELOG.md: {e}"))?;

    // 3. Refresh Cargo.lock so the workspace-member versions match.
    let status = Command::new("cargo")
        .args(["update", "-p", "deepeval-rs", "-p", "deepeval", "-p", "release"])
        .status()
        .map_err(|e| format!("failed to run cargo update: {e}"))?;
    if !status.success() {
        return Err("`cargo update` failed".into());
    }

    // 4. Commit the version bump.
    git(&[
        "add",
        "Cargo.toml",
        "CHANGELOG.md",
        "Cargo.lock",
        "crates/deepeval/Cargo.toml",
    ])?;
    git(&["commit", "-m", &format!("chore: release {tag}")])?;

    // 5. Tag and push (the tag triggers the release workflow).
    git(&["tag", "-a", &tag, "-m", &tag])?;
    git(&["push", "origin", &branch])?;
    git(&["push", "origin", &tag])?;

    println!("\nReleased {tag} and pushed to origin/{branch}.");
    println!("The GitHub Actions release workflow is now running.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bump_keywords() {
        assert_eq!(parse_bump("major").unwrap(), Bump::Major);
        assert_eq!(parse_bump("minor").unwrap(), Bump::Minor);
        assert_eq!(parse_bump("patch").unwrap(), Bump::Patch);
        assert_eq!(parse_bump("1.2.3").unwrap(), Bump::Exact("1.2.3".into()));
        assert!(parse_bump("bogus").is_err());
        assert!(parse_bump("1.2").is_err());
    }

    #[test]
    fn computes_next_versions() {
        assert_eq!(next_version("0.1.0", &Bump::Major).unwrap(), "1.0.0");
        assert_eq!(next_version("0.1.0", &Bump::Minor).unwrap(), "0.2.0");
        assert_eq!(next_version("0.1.0", &Bump::Patch).unwrap(), "0.1.1");
        assert_eq!(next_version("1.2.3", &Bump::Exact("2.0.0".into())).unwrap(), "2.0.0");
        assert!(next_version("not-a-version", &Bump::Patch).is_err());
    }

    #[test]
    fn reads_and_sets_workspace_version() {
        let toml = "[workspace]\nresolver = \"2\"\n\n[workspace.package]\nversion = \"0.1.0\"\nedition = \"2021\"\n";
        assert_eq!(read_workspace_version(toml).unwrap(), "0.1.0");
        let updated = set_workspace_version(toml, "0.2.0");
        assert_eq!(read_workspace_version(&updated).unwrap(), "0.2.0");
        assert!(updated.contains("version = \"0.2.0\""));
    }

    #[test]
    fn sets_dep_version() {
        let manifest = "[dependencies]\ndeepeval-rs = { path = \"../deepeval-rs\", features = [\"cli\"] }\ntokio.workspace = true\n";
        let updated = set_dep_version(manifest, "deepeval-rs", "0.2.0");
        assert!(updated.contains("deepeval-rs = { path = \"../deepeval-rs\", features = [\"cli\"], version = \"0.2.0\" }"));
        // A second run updates the existing version rather than duplicating it.
        let updated2 = set_dep_version(&updated, "deepeval-rs", "0.3.0");
        assert_eq!(updated2.matches("version = \"0.3.0\"").count(), 1);
        assert!(updated2.contains("version = \"0.3.0\""));
    }

    #[test]
    fn moves_unreleased_into_new_section() {
        let changelog = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- thing\n\n## [0.1.0] - 2026-09-06\n\nInitial.\n";
        let updated = update_changelog(changelog, "0.2.0", "2026-09-08").unwrap();
        assert!(updated.contains("## [Unreleased]"));
        assert!(updated.contains("## [0.2.0] - 2026-09-08"));
        assert!(updated.contains("### Added\n\n- thing"));
        // The old section must still be present.
        assert!(updated.contains("## [0.1.0] - 2026-09-06"));
        // The body must not appear twice.
        assert_eq!(updated.matches("- thing").count(), 1);
    }
}
