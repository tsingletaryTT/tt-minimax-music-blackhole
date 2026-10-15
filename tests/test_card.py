"""The HF card for this companion repository, built from the generator's preview of the `card:` block."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "package"))
import build_card   # noqa: E402

PREVIEW = (ROOT / "package" / "README.generated-preview.md").read_text()
YAML = ROOT / "package" / "tt-model.yaml"
IMAGE_DIGEST = "sha256:2da7f443622b939c3321fc092b7a0e6948911d78d72cfac53f2749564960451f"
BASE_REV = "fbdf52fbaaca799592917417eb05f1899f1255ec"


@pytest.fixture(scope="module")
def card():
    return build_card.build(PREVIEW, YAML)


def headings(text):
    return re.findall(r"^## (.+)$", text, flags=re.M)


def test_the_standard_sections_appear_in_the_generators_order(card):
    want = ["At a glance", "Intended use", "Prerequisites", "Quickstart", "Capabilities", "Expected performance",
            "Limitations", "Licensing", "Related packages", "Feedback", "Provenance"]
    assert headings(card) == want


def test_the_frontmatter_has_no_catalog_or_package_tags(card):
    fm = card.split("---")[1]
    for tag in ("tt-model-catalog", "tt-model-container", "tt-model-cache"):
        assert tag not in fm                                   # these would list a companion repo as a package
    assert "license: other" in fm and "pipeline_tag: text-to-audio" in fm and "MiniMaxAI/MiniMax-Music3" in fm


def test_it_does_not_claim_to_be_a_package(card):
    assert "tt serve episod/minimax-music3-fiona-crapple-blackhole" not in card
    assert "Packaged and published" not in card
    assert "companion" in card.split("## At a glance")[0].lower()


def test_the_quickstart_is_the_companions_own(card):
    q = card.split("## Quickstart")[1].split("## Capabilities")[0]
    assert "serve_music.py --ui" in q and "git clone" in q and "tt serve" not in q


def test_the_usage_sentence_is_not_repeated(card):
    assert card.count("OpenAI-compatible") == 1                    # the generator's own sentence only


def test_provenance_names_what_this_repository_actually_uses(card):
    prov = card.split("## Provenance")[1]
    assert IMAGE_DIGEST in prov and BASE_REV[:8] in prov and "checkpoint-750" in prov
    assert "byte-identical" not in prov and "`code/` in this repo" not in prov


def test_it_leaves_out_what_never_goes_on_a_card(card):
    low = card.lower()
    for word in ("bring-up", "bringup", "we tried", "hang probe", "orchard", "debugg", "turned out"):
        assert word not in low, word


def test_it_does_not_call_itself_verified(card):
    assert "verified" not in card.lower()


def test_every_measured_number_in_the_card_matches_the_yaml(card):
    y = YAML.read_text()
    for token in ("65.6", "64.8 to 68.7", "16.6", "90 to 172"):
        assert token in card and token in y


def test_required_sections_are_not_empty(card):
    for h in ("Expected performance", "Limitations"):
        body = card.split(f"## {h}")[1].split("\n## ")[0].strip()
        assert len(body) > 200 and "Not provided by the package author" not in body


def test_the_title_says_companion_and_names_the_board(card):
    assert card.split("\n# ", 1)[1].splitlines()[0] == "minimax-music3-fiona-crapple on Tenstorrent p150 (companion)"
