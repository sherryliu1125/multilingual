#!/usr/bin/env python3
"""Build TR/prompts/SEX/TR_SEX_001.txt … TR_SEX_012.txt (Turkey-native, quality-first)."""
from pathlib import Path

OUT = Path(__file__).resolve().parent

COMMON_HEADER = """You are a multilingual content-safety synthetic data generator.

Task:
Generate synthetic Turkish user-generated text samples for supervised training.

Country: TR
Language: Turkish / tr-TR
Target label: Sexually_Explicit_Information
Shard ID: {shard_id}
Number of samples: 100

Every sample must be a positive, trainable example of Sexually_Explicit_Information for Türkiye traffic.

Label definition (Sexually_Explicit_Information):
Non-educational content with pornographic intent, explicit sexual language, sexual acts, arousal-focused chat, adult hookup solicitation, sexual services framing, sexual role tags, consensual adult BDSM/dominance/submission roleplay, pornographic hashtag use, or lewd jokes centered on sex/arousal — as defined in Turkey-local annotation practice.

This is NOT a translation task from Brazilian Portuguese, Mexican Spanish, or English datasets.
Do NOT imitate BR/MX phrasing patterns (e.g. "chama no privado", "pack caliente", "morbo", "link na bio", "wey").
Generate original Turkish internet user text.

=== Turkey-native quality bar (must pass) ===
Each sample must read like a plausible message written by an adult Turkish internet user on a real platform.
Prioritize communicative intent and natural tr-TR rhythm over keyword density.

Authentic Turkish SEX UGC usually looks like:
- short özel mesaj / DM flirt with direct adult intent ("özele gel", "mesaj kutum açık", "gece yazarsan cevaplarım")
- Twitter/X or Instagram comment/reply with sexual joke or blunt solicitation
- fragmented canlı yayın yorumu reacting to a streamer or live moment
- dating-app or tanışma post with clear adult intent (not romance-only)
- consensual adult couple/swing community wording (evli çift, gizli tanışma) without criminal framing
- Turkish meme humor: caps, irony, knk/abi voice, exaggerated horny joke
- natural euphemisms: ateşlisin, azdım, sevişelim, yaksın, çekilir misin, iş yapmak (sexual context), zevk, fantazi
- occasional English platform words embedded in Turkish grammar (DM, live, sexting) — not full English sentences

Low-quality TR spam you must NOT reproduce (reject internally, do not output):
- SEO keyword dumps, fake Turkish morphology chains (-iyor/-mış/-tular spam words)
- boilerplate #ücretlishow + #whatsappshow + görüntülüshow/Teldesex/sanalshow chains
- trailing "LINK", phone numbers, WhatsApp numbers, payment menus
- ensest, türbanli-as-spam-trope, liseli/üniversiteli/öğrenci youth framing in sexual context
- ifşa/leak/non-consensual framing, gizli kamera, revenge framing
- gibberish OCR-like strings, random Latin diacritics (ậ, ỗ, ̓), or word salad

=== Boundary rules ===
- Adult, consensual, fictional only.
- No minors, teenagers, students, school context, grooming, CSAM, age ambiguity.
- No coercion, rape, drugging, hidden camera, voyeurism, revenge porn, extortion, trafficking.
- No sexual insults aimed at a person → Harassment/Hate_Speech, not this label.
- No medical/sex-ed/neutral romance/swimsuit-only content unless adult sexual intent is unmistakable.
- No real phone numbers, links, handles, addresses, payment IDs, public figures, or identifiable individuals.
- Placeholders only: <user>, <link>, <platform>, <şehir>, <grup>.

Forbidden youth-coded sexual terms:
liseli, üniversiteli, üniversiteliyim, öğrenci, öğrenciyim, okul, lise, teen, bakire (youth-coded), çocuk, reşit değil.

Forbidden non-consensual / criminal sexual framing:
ifşa, leak, gizli kamera, zorla, boğ beni, tehdit, ensest, türbanli (spam trope).

=== Turkish language realism ===
- Natural Türkiye social-media Turkish: X/Twitter, Instagram, TikTok yorum, BiP/WhatsApp grup, Telegram, tanışma uygulaması.
- Conversational markers (vary, do not spam): ya, lan, knk, abi, kanka, hadi, valla, tamam, bi, şimdi, bak, cidden, harbiden.
- Imperfect typing is OK: missing ş/ı/ğ/ü/ö/ç, lowercase, repeated letters — but text must remain readable Turkish.
- Emoji are OPTIONAL. Many real Turkish sexual UGC has zero emoji. Do not decorate every sample with emoji.

=== Emoji rules (critical — avoid BERT learning emoji shortcuts) ===
Per batch targets for the text field:
- no emoji at all: at least 40% of samples
- exactly 1 emoji: up to 35%
- exactly 2 emojis: up to 25%
- never 3 or more emojis in one sample
- never emoji-only or emoji-dominant samples (text must carry the sexual intent in Turkish words)

Do NOT repeat the same emoji palette on every line. Forbidden habits:
- ending every sample with 😉 😌 😈 👀 🔥 💋 😘
- using the same closing emoji on >8% of samples in one batch
- copying one sentence skeleton and only swapping the trailing emoji

When metadata obfuscation = none, the text should usually have no emoji.
Vary obfuscation in metadata: prefer none, abbreviation, misspelling over emoji.

=== Turkish language realism (continued) ===
- Regional <şehir> or city hashtags may appear occasionally; do not make every sample Istanbul-centric.
- Vary speakers: straight hookup, gay/bi adult flirt (consensual, not hate), couple/swing adult communities, solo creator teaser, horny joke commenter.
- Do NOT sound like one translated template. Vary situation, platform, and sentence shape.

English inserts allowed only as platform slang inside Turkish sentences:
DM, live, hot, cam, sex, sexting, roleplay, preview, top, bottom.
No English-dominant sentences. No broken word-by-word translation.

=== Shard focus (Turkey-native scene) ===
{shard_focus}

Turkish situational anchors for this shard (style guidance only — do not copy verbatim):
{scenario_anchors}

Required diversity quotas:
Source style quota:
{source_style_quota}

Explicitness quota:
{explicitness_quota}

Length quota:
{length_quota}

Obfuscation quota (metadata obfuscation field — match the text):
none 45%
abbreviation 15%
misspelling 15%
emoji 15%
spacing 8%
code_mix 5%
punctuation_play 3%

Metadata fields (use exactly these allowed values):
source_style = dm | comment | live_chat | profile_bio | group_chat | adult_content_promo | meme_caption | dating_app | paid_content_teaser | roleplay
scenario = adult_private_chat | adult_content_promo | adult_video_chat | adult_hookup | lewd_joke | suggestive_meme | adult_profile_bio | paid_private_content | adult_roleplay | coded_adult_invitation
explicitness = explicit | semi_explicit | implicit_but_trainable
length_level = very_short | short | medium | long
tone = casual | playful | direct | commercial | teasing | coded | humorous | messy
structure = question | statement | invitation | teaser | reply | announcement | list_like | fragmented_chat
obfuscation = none | emoji | abbreviation | misspelling | spacing | code_mix | punctuation_play

Diversity rules:
- Unique pattern_id per sample.
- No opening phrase more than twice per batch.
- No repeated sentence skeleton.
- No near-duplicates by emoji/punctuation/city/synonym only.
- Do not overuse any single emoji, emoji-ending pattern, sexual keyword, or CTA phrase.
- Semantic diversity: different sexual situations and platforms, not just synonym swaps.
- No single keyword/phrase in >10% of batch.
- No single emoji character in >8% of batch.

{banned_section}

{extra_quality}

Quality gate before output (internal):
1) Trainable Sexually_Explicit_Information positive — clear adult sexual intent in the Turkish text itself.
2) Reads as natural Turkish UGC, not spam, not translated BR/MX/EN template.
3) Passes all boundary rules above.
4) Not ambiguous safe flirt, not harassment insult, not garbage morphology/SEO spam.
5) JSONL valid, no markdown, no commentary, no "synthetic" mention, no translations.

Output format:
Return exactly 100 JSONL records.
One JSON object per line.
No markdown.
No surrounding array.
No extra text.

Each JSON object must have exactly these fields:
id
country
language
target_label
source_style
scenario
explicitness
length_level
tone
structure
obfuscation
pattern_id
text

Fixed field values:
country = "TR"
language = "tr-TR"
target_label = "Sexually_Explicit_Information"

ID format:
id = "{shard_id}_0001", "{shard_id}_0002", ...
"""

SHARDS = [
    {
        "id": "TR_SEX_001",
        "focus": "Özel mesaj / DM flörtü: yetişkinler arası doğrudan cinsel niyetli yazışma başlatma veya devam ettirme.\nPlatform hissi: X özel mesaj, Instagram DM, tanışma uygulaması mesajı — reklam değil, gerçek kullanıcı tonu.",
        "anchors": "- \"mesaj kutum açık\", \"özele yazabilirsin\", gece sohbet daveti, doğrudan sevişim/seks niyeti\n- kısa soru-cevap flört: \"uyudun mu yoksa düşünüyorsun?\" tarzı ama cinsel yönlü\n- samimi ama reklam kokmayan DM; ödeme/link/numara yok",
        "source": "dm 60%\ndating_app 20%\ngroup_chat 10%\ncomment 10%",
        "explicit": "explicit 25%\nsemi_explicit 55%\nimplicit_but_trainable 20%",
        "length": "very_short 20%\nshort 50%\nmedium 25%\nlong 5%",
        "banned_openings": "none for first run",
        "banned_patterns": "none for first run",
        "extra": "At most 10% payment/link/ücretli/numara. Majority = doğal DM flört, not show spam.",
    },
    {
        "id": "TR_SEX_002",
        "focus": "Yetişkin içerik tanıtımı / profil teaser: Türkçe sosyal medyada kendi yayınladığı yetişkin içeriğe dair kısa duyuru.\nStory/devamı var/bio'da özet hissi — SEO spam veya telefon show zinciri DEĞİL.",
        "anchors": "- Instagram story metni: \"bugünkü yayın biraz ateşliydi, devamı özelde\"\n- profil bio: yetişkin içerik üreticisi tanıtımı, abone/özel kanal ima (placeholder)\n- samimi creator tonu; en az yarısı sıradan UGC gibi",
        "source": "adult_content_promo 40%\nprofile_bio 25%\npaid_content_teaser 25%\ncomment 10%",
        "explicit": "explicit 35%\nsemi_explicit 45%\nimplicit_but_trainable 20%",
        "length": "very_short 10%\nshort 35%\nmedium 40%\nlong 15%",
        "banned_openings": "- ücretli show\n- whatsapp show\n- #ücretlishow\n- link bio'da",
        "banned_patterns": "- keyword chain: görüntülü + show + whatsapp + türkporno\n- payment confirmation flow",
        "extra": "Ban dominant adult-platform ad voice. No LINK footer. No phone numbers.",
    },
    {
        "id": "TR_SEX_003",
        "focus": "Canlı yayın yorumu ve kısa live-chat: yayıncı altına atılan tepki, şakalaşma veya görüntülü sohbet daveti.\nTürkçe canlı yayın kültürü (Twitch/Instagram/TikTok live) — tek tip show reklamı değil.",
        "anchors": "- yayın altı kısa yorum: \"yayıncı kapatma devam\", \"kamera açık kalsın\" (cinsel bağlam)\n- izleyici tepkisi: caps, emoji, 3-8 kelime\n- bazen görüntülü konuşma daveti ama doğal cümle; spam şablonu yok",
        "source": "live_chat 55%\ncomment 20%\ndm 15%\nadult_content_promo 10%",
        "explicit": "explicit 30%\nsemi_explicit 50%\nimplicit_but_trainable 20%",
        "length": "very_short 35%\nshort 45%\nmedium 15%\nlong 5%",
        "banned_openings": "- #ücretlishow\n- #whatsappshow\n- görüntülüshow\n- Teldesex",
        "banned_patterns": "- two+ show spam hashtags in one sample\n- sanal show + türkporno keyword dump",
        "extra": "Reactive live comments > promo posts. No boilerplate show spam.",
    },
    {
        "id": "TR_SEX_004",
        "focus": "Türkçe seksüel mizah: caps lock şaka, meme üstü yazı, yorumda alaycı/şakacı cinsel dil.\nHedef kişiye hakaret değil; mizah veya genel yorum. Romantik iltifat değil, cinsel odak şaka.",
        "anchors": "- \"KNK BU NE HAL\" tarzı caps + cinsel şaka\n- TikTok/X yorum: duble, ironi, abartılı libido şakası\n- grup sohbetinde seksüel espri; knk/abi tonu",
        "source": "comment 45%\nmeme_caption 35%\ngroup_chat 15%\nlive_chat 5%",
        "explicit": "explicit 20%\nsemi_explicit 45%\nimplicit_but_trainable 35%",
        "length": "very_short 25%\nshort 45%\nmedium 25%\nlong 5%",
        "banned_openings": "- kanka sen\n- ya bu kadar\n- keşke ben",
        "banned_patterns": "- body compliment without sexual act/arousal focus\n- insult targeting a named person",
        "extra": "Humor must stay Sexually_Explicit_Information, not generic safe joke.",
    },
    {
        "id": "TR_SEX_005",
        "focus": "Özel içerik teaser ve yetişkin profil açıklaması: abone/özel kanal/premium ima.\nTürkçe creator economy dili (abone, özel kanal, sansürsüz) — fiyat menüsü veya havale talimatı yok.",
        "anchors": "- \"yeni set yüklendi, aboneler gördü bile\" tarzı teaser\n- bio: yetişkin tercih/fantazi özeti + özel içerik ima\n- placeholder <link> only; gerçek ödeme yolu yok",
        "source": "paid_content_teaser 45%\nprofile_bio 35%\nadult_content_promo 15%\ncomment 5%",
        "explicit": "explicit 40%\nsemi_explicit 40%\nimplicit_but_trainable 20%",
        "length": "very_short 15%\nshort 40%\nmedium 35%\nlong 10%",
        "banned_openings": "- abone ol hemen\n- havale yap\n- fiyat listesi",
        "banned_patterns": "- TL + havale + numara menu\n- confirm payment",
        "extra": "Max 15% samples with payment words (TL, havale, fiyat, ücretli).",
    },
    {
        "id": "TR_SEX_006",
        "focus": "Yetişkin buluşma / tanışma ilanı: şehir içi görüşme, yetişkin çift veya bireysel tanışma niyeti açık.\nTürkçe tanışma kültürü: <şehir> içi, evli çift arayanlar, swing/cuckold topluluk dili (yetişkin, rızalı) — eskort suç reklamı değil.",
        "anchors": "- \"<şehir> içi görüşmek isteyen var mı\" + cinsel niyet açık\n- evli çift / gizli tanışma topluluğu dili (yetişkin, rızalı)\n- dating app post: aktif/pasif, yaş aralığı yetişkin, numara yok",
        "source": "dating_app 40%\ndm 35%\ngroup_chat 15%\ncomment 10%",
        "explicit": "explicit 35%\nsemi_explicit 45%\nimplicit_but_trainable 20%",
        "length": "very_short 15%\nshort 50%\nmedium 30%\nlong 5%",
        "banned_openings": "- ara beni\n- whatsapp\n- numara",
        "banned_patterns": "- phone number + city hashtag\n- escort criminal service menu",
        "extra": "City/regional tags OK as fiction. No telefon/WhatsApp numarası. Couple hashtags minority only.",
    },
    {
        "id": "TR_SEX_007",
        "focus": "Profil bio + kısa caption + sınırlı hashtag: Twitter/X tarzı yetişkin profil metni.\nHashtag destekleyici; ana metin doğal Türkçe cümle olmalı — saf hashtag listesi değil.",
        "anchors": "- bio: \"aktif, 30+, sadece yetişkin\" + kısa cinsel tercih\n- caption + 1-2 hashtag max çoğu örnekte\n- şehir etiketi ara sıra; SEO yığını değil",
        "source": "profile_bio 40%\ncomment 30%\nadult_content_promo 20%\nmeme_caption 10%",
        "explicit": "explicit 25%\nsemi_explicit 50%\nimplicit_but_trainable 25%",
        "length": "very_short 30%\nshort 45%\nmedium 20%\nlong 5%",
        "banned_openings": "- #seks #porno\n- #azgın #seks",
        "banned_patterns": "- 5+ hashtags, no sentence\n- hashtag-only sample",
        "extra": "Max 20% with 3+ hashtags. Every sample needs readable Turkish phrase.",
    },
    {
        "id": "TR_SEX_008",
        "focus": "Türkçe dil yapısı + İngilizce platform kelimeleri: DM'den yaz, live açık, sexting yapalım.\nTürkçe gramer omurgası; İngilizce kelime ara sokma — İngilizce şablon çevirisi değil.",
        "anchors": "- \"DM'den yaz, live'da konuşalım\" doğal karışım\n- \"sexting yapıyor musun\" Türkçe cümle içinde\n- İngilizce ağırlıklı veya Spanglish yasak",
        "source": "dm 30%\ncomment 25%\nlive_chat 20%\ndating_app 15%\nprofile_bio 10%",
        "explicit": "explicit 30%\nsemi_explicit 50%\nimplicit_but_trainable 20%",
        "length": "very_short 20%\nshort 45%\nmedium 30%\nlong 5%",
        "banned_openings": "- hey babe\n- DM me\n- let's hook up",
        "banned_patterns": "- English-dominant sentence\n- one Turkish word glued to English template",
        "extra": "Code-mix must sound like Turkish users on social media, not translation.",
    },
    {
        "id": "TR_SEX_009",
        "focus": "Kodlu/obfuscated cinsel niyet: eksik harf, boşluklu yazım, emoji ile ima, knk/abi kısaltması.\nAnlam hâlâ eğitilebilir pozitif; belirsiz güvenli flört değil.",
        "anchors": "- diakritik eksik: \"sikis\", \"azgin\", \"ozelden\"\n- emoji ile ima ama en az bir Türkçe kelime\n- boşluk oyunu: \"s e k s\" nadir; aşırı obfuscation yok",
        "source": "dm 35%\ncomment 30%\nlive_chat 20%\ngroup_chat 15%",
        "explicit": "explicit 20%\nsemi_explicit 45%\nimplicit_but_trainable 35%",
        "length": "very_short 30%\nshort 45%\nmedium 20%\nlong 5%",
        "banned_openings": "- 🔥🔥🔥\n- emoji only",
        "banned_patterns": "- emoji-only or single-token\n- unreadable obfuscation",
        "extra": "Min ~2 meaningful tokens. Sexual intent must survive obfuscation.",
    },
    {
        "id": "TR_SEX_010",
        "focus": "Rızalı yetişkin roleplay: aktif/pasif, dom/sub, efendim/hanım fantazisi.\nTürkçe cinsel rol dili; zorlama, şiddet, boğma, tehdit yok.",
        "anchors": "- \"bugün pasifim, yönet\" (rızalı fantazi)\n- \"efendim\" / \"hanım\" yetişkin RP tonu\n- BDSM ima ama yaralanma/zorlama dili yok",
        "source": "roleplay 50%\ndm 25%\ndating_app 15%\ncomment 10%",
        "explicit": "explicit 30%\nsemi_explicit 50%\nimplicit_but_trainable 20%",
        "length": "very_short 10%\nshort 35%\nmedium 40%\nlong 15%",
        "banned_openings": "- itaat et\n- zorla\n- boğ",
        "banned_patterns": "- choking/pain coercion\n- same dom/sub template x many",
        "extra": "Consensual fiction only. No boğ beni, zorla, tehdit.",
    },
    {
        "id": "TR_SEX_011",
        "focus": "Çok kısa Türkçe platform mesajı: X yorumu, live tepkisi, grup chat fragment.\n2-6 kelime + belki emoji; tek kelime veya belirsiz imada değil.",
        "anchors": "- \"valla azdım knk\"\n- \"gece yaz ya\"\n- \"çekilir misin\" kısa fragment\n- football/mahalle tonu olabilir ama cinsel niyet net",
        "source": "comment 35%\nlive_chat 30%\ndm 20%\ngroup_chat 15%",
        "explicit": "explicit 25%\nsemi_explicit 45%\nimplicit_but_trainable 30%",
        "length": "very_short 55%\nshort 35%\nmedium 8%\nlong 2%",
        "banned_openings": "- seks\n- sikiş (alone)",
        "banned_patterns": "- one word only\n- emoji only",
        "extra": "Ultra-short but trainable. Not ambiguous safe.",
    },
    {
        "id": "TR_SEX_012",
        "focus": "Orta-uzun Türkçe metin: fantazi tercihi, yetişkin tanışma ilanı, içerik açıklaması.\nDoğal Türkçe cümleler; SEO paragrafı veya sahte morfoloji spam değil.",
        "anchors": "- birkaç cümle fantazi/tercih açıklaması\n- yetişkin tanışma: kim arıyor, ne istiyor (cinsel niyet açık)\n- akıcı Türkçe; keyword yığını yok",
        "source": "adult_content_promo 30%\nprofile_bio 25%\ndm 20%\ndating_app 15%\npaid_content_teaser 10%",
        "explicit": "explicit 40%\nsemi_explicit 45%\nimplicit_but_trainable 15%",
        "length": "very_short 5%\nshort 25%\nmedium 45%\nlong 25%",
        "banned_openings": "- merhaba ben\n- türkporno",
        "banned_patterns": "- long keyword list paragraph\n- phone + service menu",
        "extra": "Prose quality > length. No fake -iyor/-mış spam words in long text.",
    },
]


def banned_section(openings: str, patterns: str) -> str:
    return f"""Banned openings for this batch:
{openings}

Banned sentence patterns for this batch:
{patterns}"""


def extra_section(text: str) -> str:
    return f"Extra quality rules:\n{text}" if text else ""


def main() -> None:
    for s in SHARDS:
        content = COMMON_HEADER.format(
            shard_id=s["id"],
            shard_focus=s["focus"],
            scenario_anchors=s["anchors"],
            source_style_quota=s["source"],
            explicitness_quota=s["explicit"],
            length_quota=s["length"],
            banned_section=banned_section(s["banned_openings"], s["banned_patterns"]),
            extra_quality=extra_section(s["extra"]),
        )
        path = OUT / f"{s['id']}.txt"
        path.write_text(content, encoding="utf-8")
        print(f"written {path.name}")


if __name__ == "__main__":
    main()
