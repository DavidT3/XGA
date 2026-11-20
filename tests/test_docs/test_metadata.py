#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 11/20/26, 8:46 AM. Copyright (c) The Contributors.
"""Tests ensuring documentation author/contributor metadata stays synchronized with .zenodo.json."""

import json
import os
import unittest
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ZENODO_PATH = os.path.join(REPO_ROOT, ".zenodo.json")
DOCS_DIR = os.path.join(REPO_ROOT, "new-docs")
MYST_YML_PATH = os.path.join(DOCS_DIR, "myst.yml")
CONTRIBUTORS_MD_PATH = os.path.join(DOCS_DIR, "contributors.md")


def _zenodo_name_to_display(zenodo_name: str) -> str:
    """
    Convert a Zenodo 'Family, Given' name string into 'Given Family' display format.

    :param str zenodo_name: The name formatted as 'Family, Given' from .zenodo.json.
    :return: The display name formatted as 'Given Family'.
    :rtype: str
    """
    parts = [part.strip() for part in zenodo_name.split(",", maxsplit=1)]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return zenodo_name.strip()


def _load_markdown_frontmatter(md_path: str) -> dict[str, Any]:
    """
    Extract and parse the YAML frontmatter block from a MyST Markdown file.

    :param str md_path: Path to the Markdown file.
    :return: Parsed YAML frontmatter dictionary.
    :rtype: dict[str, Any]
    """
    with open(md_path, encoding="utf-8") as md_file:
        content = md_file.read()

    if not content.startswith("---\n"):
        raise AssertionError(f"Expected YAML frontmatter at the start of {md_path}")

    end_idx = content.find("\n---\n", 4)
    if end_idx == -1:
        raise AssertionError(f"Could not find closing YAML frontmatter delimiter in {md_path}")

    frontmatter_raw = content[4:end_idx]
    parsed: dict[str, Any] = yaml.safe_load(frontmatter_raw)
    return parsed


class TestDocsMetadataConsistency(unittest.TestCase):
    """
    Verify that author and contributor definitions in new-docs/myst.yml and new-docs/contributors.md
    are consistent with the primary source of truth in .zenodo.json.
    """

    zenodo_data: dict[str, Any]
    myst_data: dict[str, Any]
    contributors_fm: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        """Load .zenodo.json, new-docs/myst.yml, and new-docs/contributors.md frontmatter once for the test class."""
        with open(ZENODO_PATH, encoding="utf-8") as zf:
            cls.zenodo_data = json.load(zf)

        with open(MYST_YML_PATH, encoding="utf-8") as mf:
            cls.myst_data = yaml.safe_load(mf)

        cls.contributors_fm = _load_markdown_frontmatter(CONTRIBUTORS_MD_PATH)

    def test_myst_yml_contributors_match_zenodo(self) -> None:
        """Ensure project.contributors and project.authors in myst.yml match .zenodo.json creators and contributors."""
        zenodo_creators = self.zenodo_data["creators"]
        zenodo_contributors = self.zenodo_data["contributors"]
        all_zenodo = zenodo_creators + zenodo_contributors

        myst_project = self.myst_data["project"]
        myst_contributors = myst_project["contributors"]
        myst_authors = myst_project["authors"]

        self.assertEqual(
            len(myst_contributors),
            len(all_zenodo),
            "Number of entries in myst.yml project.contributors does not match .zenodo.json creators + contributors.",
        )

        contributor_ids: list[str] = []
        for zen_entry, myst_entry in zip(all_zenodo, myst_contributors, strict=True):
            expected_name = _zenodo_name_to_display(zen_entry["name"])
            self.assertIn("id", myst_entry, f"Missing 'id' for contributor {expected_name} in myst.yml.")
            contributor_ids.append(myst_entry["id"])

            self.assertEqual(myst_entry.get("name"), expected_name)
            self.assertEqual(myst_entry.get("orcid"), zen_entry.get("orcid"))
            self.assertTrue(
                isinstance(myst_entry.get("roles"), list) and len(myst_entry["roles"]) > 0,
                f"Expected non-empty CRediT 'roles' list for {expected_name} in myst.yml.",
            )

        self.assertEqual(
            len(contributor_ids),
            len(set(contributor_ids)),
            "Duplicate contributor IDs found in myst.yml project.contributors.",
        )

        # project.authors in myst.yml should be the IDs of the creators from .zenodo.json in order
        expected_creator_ids = contributor_ids[: len(zenodo_creators)]
        self.assertEqual(
            myst_authors,
            expected_creator_ids,
            "myst.yml project.authors does not match the creator IDs corresponding to .zenodo.json creators.",
        )

    def test_contributors_md_matches_zenodo_and_myst_yml(self) -> None:
        """Ensure contributors.md frontmatter matches .zenodo.json (names, ORCIDs, affiliations) and myst.yml."""
        all_zenodo = self.zenodo_data["creators"] + self.zenodo_data["contributors"]
        myst_contributors = self.myst_data["project"]["contributors"]
        md_authors = self.contributors_fm.get("authors", [])

        self.assertEqual(
            len(md_authors),
            len(all_zenodo),
            "Number of authors in contributors.md does not match .zenodo.json creators + contributors.",
        )

        for zen_entry, myst_entry, md_entry in zip(all_zenodo, myst_contributors, md_authors, strict=True):
            expected_name = _zenodo_name_to_display(zen_entry["name"])
            expected_affiliations = [aff.strip() for aff in zen_entry["affiliation"].split(";") if aff.strip()]

            self.assertEqual(md_entry.get("name"), expected_name)
            self.assertEqual(md_entry.get("orcid"), zen_entry.get("orcid"))
            self.assertEqual(md_entry.get("affiliations"), expected_affiliations)

            # Ensure roles, email, and corresponding flags stay synchronized between myst.yml and contributors.md
            self.assertEqual(md_entry.get("roles"), myst_entry.get("roles"))
            self.assertEqual(md_entry.get("email"), myst_entry.get("email"))
            self.assertEqual(md_entry.get("corresponding"), myst_entry.get("corresponding"))


if __name__ == "__main__":
    unittest.main()
