from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.services.glm_ollama_ocr import GlmOllamaOcrResult


_FIXTURE_WAIT_SECONDS = 5.0


@dataclass(frozen=True)
class TestSemanticMention:
    surface: str
    canonical_label: str
    ent_type: str
    reason: str
    confidence: float = 0.99
    method: str = "fixture:semantic_entity_override"


@dataclass(frozen=True)
class TestOcrOverride:
    sha256: str
    text: str
    model_used: str = "glm-ocr:latest"
    wait_seconds: float = _FIXTURE_WAIT_SECONDS
    semantic_mentions: tuple[TestSemanticMention, ...] = ()


_FIXTURE_OVERRIDES: dict[str, TestOcrOverride] = {}


def _register_fixture(
    sha256: str,
    text: str,
    *,
    semantic_mentions: tuple[TestSemanticMention, ...] = (),
) -> None:
    _FIXTURE_OVERRIDES[sha256] = TestOcrOverride(
        sha256=sha256,
        text=text.strip(),
        semantic_mentions=semantic_mentions,
    )


_register_fixture(
    "62ad3e357c6120bc7845096a403db994d397641f6dde9598310262273391283a",
    """
receorl. porstoel. anns oxan. Furata
est mulissi. Furatũ est mancipiũ. Et
cet̃a. þaporo þege nymað onp̃tqritum.
iri. ⁊ nofdon at pruman. þone .u. a
mo. amaui. þahabbað hpilon ƿinco
pam. ꝥyspanunga onðam oðrũ hade.
iondam þridoan. amaui amauisti.
ł amasti. hisr yꝯse .ui. aƿege. amauis
tis. ł amartis. amaufriunt. łamaruN.
ealspa neo. ⁊c fpinne. noui. ic span.
neuisti. ł nesti. duspunne. neuistis.
ł nestis. gespunnon. usussuint. ł ner̃t.
hi spunnon achit nebyðna sƿa gif
reti. byð at fruman. onðamƿonde.
lauio. icdpea . laui. lauirti. þune miht
naeƿeðan hai larti .dt .ii.c̃ionctioni.
e oðqr coniugatio yspul eað enceƿe.
forðanðe aelcðara poroa þege en
dad̾ on .eo. ⁊eoðq had ones. ys þare
odre geðeodnysse. doceo iclare. do
ces. þu loerist. docet. hȩ larð. Et płr . do
cemus pecaecað. docłtis getcecað. do
""",
    semantic_mentions=(
        TestSemanticMention(
            surface="c̃ionctioni",
            canonical_label="coniunctioni",
            ent_type="GRAMMATICAL_TERM",
            reason="curated semantic concept for old english fixture; grammatical terminology",
        ),
        TestSemanticMention(
            surface="coniugatio",
            canonical_label="coniugatio",
            ent_type="GRAMMATICAL_TERM",
            reason="curated semantic concept for old english fixture; grammatical terminology",
        ),
        TestSemanticMention(
            surface="amaui",
            canonical_label="amaui",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="amauisti",
            canonical_label="amauisti",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="amartis",
            canonical_label="amastis",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="amaufriunt",
            canonical_label="amauerunt",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="noui",
            canonical_label="noui",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="neuisti",
            canonical_label="neuisti",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="doceo",
            canonical_label="doceo",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="do\nces",
            canonical_label="doces",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="docet",
            canonical_label="docet",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="do\ncemus",
            canonical_label="docemus",
            ent_type="VERB_PARADIGM_EXAMPLE",
            reason="curated semantic concept for old english fixture; verb paradigm example",
        ),
        TestSemanticMention(
            surface="ic",
            canonical_label="ic",
            ent_type="GLOSSING_LANGUAGE_ELEMENT",
            reason="curated semantic concept for old english fixture; glossing language element",
        ),
        TestSemanticMention(
            surface="þu",
            canonical_label="þu",
            ent_type="GLOSSING_LANGUAGE_ELEMENT",
            reason="curated semantic concept for old english fixture; glossing language element",
        ),
        TestSemanticMention(
            surface="hȩ",
            canonical_label="he",
            ent_type="GLOSSING_LANGUAGE_ELEMENT",
            reason="curated semantic concept for old english fixture; glossing language element",
        ),
        TestSemanticMention(
            surface="hi",
            canonical_label="hi",
            ent_type="GLOSSING_LANGUAGE_ELEMENT",
            reason="curated semantic concept for old english fixture; glossing language element",
        ),
        TestSemanticMention(
            surface="oxan",
            canonical_label="oxan",
            ent_type="LEXICAL_EXAMPLE",
            reason="curated semantic concept for old english fixture; lexical example",
        ),
        TestSemanticMention(
            surface="mancipiũ",
            canonical_label="mancipium",
            ent_type="LEXICAL_EXAMPLE",
            reason="curated semantic concept for old english fixture; lexical example",
        ),
        TestSemanticMention(
            surface="mulissi",
            canonical_label="mulieris",
            ent_type="LEXICAL_EXAMPLE",
            reason="curated semantic concept for old english fixture; lexical example",
        ),
        TestSemanticMention(
            surface="fruman",
            canonical_label="fruman",
            ent_type="MORPHOLOGICAL_EXPLANATION",
            reason="curated semantic concept for old english fixture; morphological explanation term",
        ),
        TestSemanticMention(
            surface="oðrũ hade",
            canonical_label="oðrum hade",
            ent_type="MORPHOLOGICAL_EXPLANATION",
            reason="curated semantic concept for old english fixture; morphological explanation term",
        ),
        TestSemanticMention(
            surface="geðeodnysse",
            canonical_label="bilingual glossing",
            ent_type="PEDAGOGICAL_FUNCTION",
            reason="curated semantic concept for old english fixture; pedagogical function",
        ),
    ),
)

_register_fixture(
    "f248cb4a92d1e1785ea1dd4e3c288b6ee8484303d3c8bd54d8c6de5728ee79f4",
    """
mẽt en autre boneᷤ cau
ses. ⁊honestes ⁊ꝑfitableᷤ.
En autre maniere ne.
lest mie nirer.⁊por ce
qui iure senz reson le
nõ nr̃e seigneur ⁊pour
noiant.se il nire faus
a son esciẽt il se ꝑuire.
⁊fet contre ceꝯmãdem̃t.
⁊peche mortieumẽt.car
il nire contre sacõciẽce
⁊est aantendre qnͣt lẽ
se ꝑnire apenseement
⁊ a delib̾acion. Mes cil qͥ
iure uoir a son escient
⁊toutes uoies por noi
ant ou pour aucune
mauuaise reson.non
mie malicieusement.
mes legierem̃t ⁊senz
blaspheme peche ueni
aument.mes la costu
mance est perilleuse ⁊
puet bien torner apechie
mortel.qui ne sen garde.
Mes cil qui iure horrible
mẽt de nr̃e seigneur. ou
des sainz ⁊ledespit ⁊depie
ce ⁊endlit blasphemes qͥ
nefont mie adire.cil pe
chent mortieumẽt ⁊ne
puet auoir reson qui lẽ
puisse escuser.⁊cil qui
plus la coustumẽt pluᷤ
utliens
pechent.
li .iii. ꝯmãdemẽz
Itierz ꝯmandemẽz
est tiex. Garde q̃ tu
saintefies le ior dou sa
mecli. Cest aclire tu ne
feras mie auiour du sa
mecli tes besoignes ⁊tes
ouraignes com tu seus
faire aus autres iours.
mes te reposeras ⁊feras
pourmieuz entendre a
prier ⁊ aseruir ton crea
teur qui se reposa au sep
tieme iour des oeures q̃
il auoit fetes les .ui. iors.
deuant. en q̃ il fist le mõ
de ⁊ordena. ¶Ce ꝯman
dement acomplist esꝑi
""",
    semantic_mentions=(
        TestSemanticMention(
            surface="nr̃e seigneur",
            canonical_label="nostre seigneur",
            ent_type="RELIGIOUS_TITLE",
            reason="curated semantic concept for french fixture; non-linkable title",
        ),
        TestSemanticMention(
            surface="sainz",
            canonical_label="sainz",
            ent_type="RELIGIOUS_GROUP",
            reason="curated semantic concept for french fixture; non-linkable religious group",
        ),
        TestSemanticMention(
            surface="sa\nmecli",
            canonical_label="samedi",
            ent_type="TEMPORAL_LITURGICAL_TERM",
            reason="curated semantic concept for french fixture; non-linkable temporal-liturgical term",
        ),
        TestSemanticMention(
            surface="blaspheme",
            canonical_label="blaspheme",
            ent_type="THEOLOGICAL_CONCEPT",
            reason="curated semantic concept for french fixture; non-linkable theological concept",
        ),
        TestSemanticMention(
            surface="peche",
            canonical_label="peche",
            ent_type="THEOLOGICAL_CONCEPT",
            reason="curated semantic concept for french fixture; non-linkable theological concept",
        ),
        TestSemanticMention(
            surface="sacõciẽce",
            canonical_label="conscience",
            ent_type="THEOLOGICAL_CONCEPT",
            reason="curated semantic concept for french fixture; non-linkable theological concept",
        ),
        TestSemanticMention(
            surface="nire faus",
            canonical_label="iurer faus",
            ent_type="MORAL_ACTION",
            reason="curated semantic concept for french fixture; non-linkable moral action",
        ),
        TestSemanticMention(
            surface="aseruir ton crea\nteur",
            canonical_label="seruir ton createur",
            ent_type="MORAL_ACTION",
            reason="curated semantic concept for french fixture; non-linkable moral action",
        ),
    ),
)

_register_fixture(
    "625b69d4299b2a2e1685648709cbf3b7c8a7d8541a4ca535d5d168ed8e7340f5",
    """
enit qui compedes diuina potñcia soluit E & hac uxore non erit sermo pudieus Inmicos multos habes ⁊ plures amicos Q sperat celat differtur pey omnia secla Pͭ ꝑter tuũ lucrõ iram pacieris amaram Fortuna bona tibi es felix temꝑe ur̃o Gencrare miserum hedem si dera donant Ninces adu̾sariũ ⁊ placito hunc suꝑabis Sopniũ ĩ cͣpula fuit. ⁊ erit uisio uana Eme nam est bona ⁊ utilis emys Est fur exͣneus cum furto longius ibit H tuns amicus erit tibi tempore gͣtus Et tuũ prosꝑum uiaq tuta manet Deciꝑe te uult. ⁊ te non diligit illa ꝑ egre profecto non datur cito reu̾ti Imfirmus ille tiuis sanatur tp̃re breui Tibi dico c̾tum reddet᷑ amissio tisa P auper cris nimis ⁊ manẽs debito semii T u tela tͥ data non est: tim̾e nec eẽt U iues ⁊ non diu ⁊ mors optata cessabit E difica plenam herit mansio tua A superis datam doctnam suscipe tuam Utilis est tibi fcã mutacio per nos C ur q̃ris honorem fug̾e tibi melius eẽt Quod optas properat festina susciꝑe uenit Et tͥ diuicie dabuntur a dño semper Huestis maliuolis tibi reddec ppołm Non erit ħ pugna cessabit altera tͣba Abaton: necosse ÷ 25 28 P rA 20 25
""",
    semantic_mentions=(
        TestSemanticMention(
            surface="Fortuna",
            canonical_label="fortuna",
            ent_type="ABSTRACT_CONCEPT",
            reason="curated semantic concept for latin fixture; abstract concept",
        ),
        TestSemanticMention(
            surface="honorem",
            canonical_label="honor",
            ent_type="ABSTRACT_CONCEPT",
            reason="curated semantic concept for latin fixture; abstract concept",
        ),
        TestSemanticMention(
            surface="doctnam",
            canonical_label="doctrina",
            ent_type="ABSTRACT_CONCEPT",
            reason="curated semantic concept for latin fixture; abstract concept",
        ),
        TestSemanticMention(
            surface="mors",
            canonical_label="mors",
            ent_type="ABSTRACT_CONCEPT",
            reason="curated semantic concept for latin fixture; abstract concept",
        ),
        TestSemanticMention(
            surface="uxore",
            canonical_label="uxor",
            ent_type="SOCIAL_ROLE",
            reason="curated semantic concept for latin fixture; social role",
        ),
        TestSemanticMention(
            surface="amicus",
            canonical_label="amicus",
            ent_type="SOCIAL_ROLE",
            reason="curated semantic concept for latin fixture; social role",
        ),
        TestSemanticMention(
            surface="Inmicos",
            canonical_label="inimicus",
            ent_type="SOCIAL_ROLE",
            reason="curated semantic concept for latin fixture; social relation",
        ),
        TestSemanticMention(
            surface="dño",
            canonical_label="domino",
            ent_type="RELIGIOUS_REFERENCE",
            reason="curated semantic concept for latin fixture; religious reference",
        ),
        TestSemanticMention(
            surface="debito",
            canonical_label="debitum",
            ent_type="MORAL_PRACTICAL_CONCEPT",
            reason="curated semantic concept for latin fixture; moral-practical concept",
        ),
        TestSemanticMention(
            surface="furto",
            canonical_label="furtum",
            ent_type="MORAL_PRACTICAL_CONCEPT",
            reason="curated semantic concept for latin fixture; moral-practical concept",
        ),
        TestSemanticMention(
            surface="pugna",
            canonical_label="pugna",
            ent_type="MORAL_PRACTICAL_CONCEPT",
            reason="curated semantic concept for latin fixture; moral-practical concept",
        ),
        TestSemanticMention(
            surface="prosꝑum",
            canonical_label="prosperum",
            ent_type="STATE_OR_FORTUNE",
            reason="curated semantic concept for latin fixture; state or fortune marker",
        ),
        TestSemanticMention(
            surface="felix",
            canonical_label="felix",
            ent_type="STATE_OR_FORTUNE",
            reason="curated semantic concept for latin fixture; state or fortune marker",
        ),
        TestSemanticMention(
            surface="festina",
            canonical_label="festina",
            ent_type="ACTION_OR_WARNING",
            reason="curated semantic concept for latin fixture; advisory action or warning",
        ),
        TestSemanticMention(
            surface="susciꝑe",
            canonical_label="suscipe",
            ent_type="ACTION_OR_WARNING",
            reason="curated semantic concept for latin fixture; advisory action or warning",
        ),
    ),
)


def get_test_ocr_fixture(image_bytes: bytes) -> TestOcrOverride | None:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return _FIXTURE_OVERRIDES.get(digest)


def get_test_ocr_override(image_bytes: bytes) -> GlmOllamaOcrResult | None:
    fixture = get_test_ocr_fixture(image_bytes)
    if fixture is None:
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            normalized = ImageOps.exif_transpose(image)
            width, height = normalized.size
    except (UnidentifiedImageError, OSError):
        width, height = 0, 0

    lines = [line for line in fixture.text.splitlines() if line.strip()]
    return GlmOllamaOcrResult(
        text=fixture.text,
        lines=lines,
        model_used=fixture.model_used,
        warnings=[],
        original_size_bytes=len(image_bytes),
        original_width=width,
        original_height=height,
        processed_width=width,
        processed_height=height,
        processed_size_bytes=len(image_bytes),
        preprocessing_applied=False,
        processed_variant_name="fixture_exact_text",
        attempts_used=1,
        duration_seconds=float(fixture.wait_seconds),
    )
