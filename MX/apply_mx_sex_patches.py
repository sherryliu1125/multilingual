#!/usr/bin/env python3
"""
Apply 5 patches to all MX SEX prompt files.

Patches 1-4 → MX_SEX_001 through MX_SEX_012
Patch 5     → MX_SEX_010 only (replaces the boundary section)

Run: python apply_mx_sex_patches.py
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "SEX"

# ─────────────────────────────────────────────────────────────
# Patch text blocks
# ─────────────────────────────────────────────────────────────

PATCH_1 = """\
Hard exclusions from Sexually_Explicit_Information:
Do NOT generate samples involving minors, youth-coded sexual framing, school context, or age ambiguity.
Forbidden in sexual contexts:
mija, morrita, morritas, chavita, chava, niña, colegiala, estudiante, escuela, prepa, secundaria, teen, barely legal.

Do NOT generate non-consensual or privacy-violating sexual content:
no leaked videos, no leaks, no "no lo subí yo", no "no sé quién lo subió", no hidden camera, no secret recording, no screenshots, no recordings, no revenge content, no private-photo requests, no nude-photo requests, no "sin dejar rastro", no "sin que se entere", no blackmail, no sexual extortion.

Do NOT generate coercive or violent sexual content:
no forced sex, no threats, no forced obedience, no choking, no breath restriction, no bozal, no sin aire, no ahogarse, no agarrar del cuello, no sin piedad, no screaming from pain, no "te obligo", no "no te voy a perdonar", no "calladita", no wrist grabbing, no pain-focused sex, no punishment framing.

If a sample would involve any of the above, do not output it for this shard."""

PATCH_2 = """\
Transaction / paid-content control:
This batch must NOT be dominated by adult-content sales, packs, payment, links, rooms, codes, passes, subscriptions, or access mechanics.

At most 10% of this batch may mention any of:
pack, pago, pesos, precio, tarjeta, SPEI, OXXO, MercadoPago, PayPal, transferencia, depósito, link, acceso, código, pass, sala, room, suscripción, descuento, oferta, descarga.

Do NOT include:
real or fake phone numbers, WhatsApp, payment handles, payment instructions, payment confirmation, "confirmar pago", "pago seguro", "te mando el link al pagar", "solo con reservación", or detailed price menus.

Adult-content promotion is allowed only as a minority style.
Most samples should be natural user-generated sexual chat, comments, DMs, jokes, live-chat fragments, or adult flirtation with clear sexual intent."""

PATCH_3 = """\
Stricter anti-duplication:
Do not generate the same idea with synonyms.
Do not reuse these sentence skeletons:
- "si te gusta X, aquí tienes Y"
- "si te late X, manda DM"
- "contenido privado/exclusivo + link/acceso"
- "pack + caliente + pago"
- "quién quiere + entrar/ver"
- "solo mayores + DM"
- "si buscas algo diferente/picante, aquí estoy"
- "aquí tienes un video donde..."
- "si no te gusta el morbo..."
- "si quieres algo caliente..."

At least 70% of the batch must NOT contain these words:
pack, link, DM, privado, contenido, caliente, morbo, acceso, pago.

No opening phrase may repeat more than once in a batch.
No sentence skeleton may repeat.
No more than 3 samples may use the same main CTA verb:
manda, pásate, entra, dime, escribe, ven, checa, avisa.
If two samples feel structurally similar, rewrite one internally before output."""

PATCH_4 = """\
Mexican Spanish naturalness:
Write like different real Mexican adult users, not like a translated dataset.
Avoid stiff promotional Spanish, corporate ad copy, Spain Spanish, Argentine/Uruguayan voseo, and broken bilingual grammar.

Do NOT use:
tenés, tenes, vos, che, boludo, tío, vale, vosotros.

Do NOT overuse:
wey/güey, jaja, al priv, manda DM, pack, morbo, caliente.

English code-mixing is allowed only when natural for Mexican social media:
DM, live, hot, preview, link, top, bottom, roleplay.
Do not produce broken Spanglish like word-by-word translation."""

# For MX_SEX_010: replaces the entire "Important boundary rules:" section
PATCH_5_BOUNDARY = """\
Important boundary rules:
For this BDSM / roleplay shard:
Generate only adult, consensual, clearly negotiated, non-injurious roleplay.
The tone can be dominant/submissive, but it must not imply real coercion, threats, pain, injury, breath restriction, fear, inability to refuse, or non-consent.

Forbidden:
bozal, sin aire, ahorcar, asfixiar, cuello, sin piedad, gritar, obligar, castigo real, calladita, no te voy a perdonar, muñecas, dolor, no puedes irte, no puedes decir que no.

Prefer:
consensual roleplay setup, fantasy framing, mutually agreed limits, playful dominant/submissive language, soft kink, adult chat negotiation, "si te late", "si ambos queremos", "con límites claros", "todo consensuado".

Do not generate violent commands or humiliation targeting a real person."""

# ─────────────────────────────────────────────────────────────
# Idempotency sentinel strings (first unique line of each patch)
# ─────────────────────────────────────────────────────────────
SENTINEL_1 = "Hard exclusions from Sexually_Explicit_Information:"
SENTINEL_2 = "Transaction / paid-content control:"
SENTINEL_3 = "Stricter anti-duplication:"
SENTINEL_4 = "Mexican Spanish naturalness:"
SENTINEL_5 = "For this BDSM / roleplay shard:"


def apply_patches_standard(content: str, shard_id: str) -> str:
    """Apply patches 1-4 to a standard (non-010) shard."""

    # ── Patch 1: insert after boundary anchor ──────────────────
    boundary_anchor = "- Use placeholders if needed: <user>, <link>, <platform>, <ciudad>, <grupo>."
    if SENTINEL_1 not in content:
        if boundary_anchor in content:
            content = content.replace(
                boundary_anchor,
                boundary_anchor + "\n\n" + PATCH_1,
                1,
            )
            print(f"  [{shard_id}] Patch 1 applied.")
        else:
            print(f"  [{shard_id}] WARNING: boundary anchor not found, Patch 1 skipped.")
    else:
        print(f"  [{shard_id}] Patch 1 already present, skipped.")

    # ── Patch 4: insert before "Shard focus:" ──────────────────
    shard_focus_marker = "\nShard focus:"
    if SENTINEL_4 not in content:
        if shard_focus_marker in content:
            content = content.replace(
                shard_focus_marker,
                "\n" + PATCH_4 + "\n" + shard_focus_marker.lstrip("\n"),
                1,
            )
            print(f"  [{shard_id}] Patch 4 applied.")
        else:
            print(f"  [{shard_id}] WARNING: 'Shard focus:' not found, Patch 4 skipped.")
    else:
        print(f"  [{shard_id}] Patch 4 already present, skipped.")

    # ── Patch 3: insert before "Banned openings for this batch:" ──
    banned_marker = "\nBanned openings for this batch:"
    if SENTINEL_3 not in content:
        if banned_marker in content:
            content = content.replace(
                banned_marker,
                "\n" + PATCH_3 + "\n" + banned_marker.lstrip("\n"),
                1,
            )
            print(f"  [{shard_id}] Patch 3 applied.")
        else:
            print(f"  [{shard_id}] WARNING: 'Banned openings' not found, Patch 3 skipped.")
    else:
        print(f"  [{shard_id}] Patch 3 already present, skipped.")

    # ── Patch 2: insert before "Quality requirements:" ──────────
    quality_marker = "\nQuality requirements:"
    if SENTINEL_2 not in content:
        if quality_marker in content:
            content = content.replace(
                quality_marker,
                "\n" + PATCH_2 + "\n" + quality_marker.lstrip("\n"),
                1,
            )
            print(f"  [{shard_id}] Patch 2 applied.")
        else:
            print(f"  [{shard_id}] WARNING: 'Quality requirements:' not found, Patch 2 skipped.")
    else:
        print(f"  [{shard_id}] Patch 2 already present, skipped.")

    return content


def apply_patches_010(content: str) -> str:
    """Apply patches 2-5 to MX_SEX_010 (patch 5 replaces boundary section)."""
    shard_id = "MX_SEX_010"

    # ── Patch 5: replace "Important boundary rules:" block ──────
    if SENTINEL_5 not in content:
        # Find start: "Important boundary rules:\n"
        start_marker = "\nImportant boundary rules:\n"
        # Find end: the blank line before "Mexican Spanish realism requirements:"
        end_marker = "\nMexican Spanish realism requirements:"
        if start_marker in content and end_marker in content:
            start_idx = content.index(start_marker)
            end_idx = content.index(end_marker)
            # Replace from start_marker up to (not including) end_marker
            old_section = content[start_idx:end_idx]
            content = content.replace(old_section, "\n" + PATCH_5_BOUNDARY + "\n", 1)
            print(f"  [{shard_id}] Patch 5 applied (boundary section replaced).")
        else:
            print(f"  [{shard_id}] WARNING: boundary section markers not found, Patch 5 skipped.")
    else:
        print(f"  [{shard_id}] Patch 5 already present, skipped.")

    # ── Patch 4: insert before "Shard focus:" ────────────────────
    shard_focus_marker = "\nShard focus:"
    if SENTINEL_4 not in content:
        if shard_focus_marker in content:
            content = content.replace(
                shard_focus_marker,
                "\n" + PATCH_4 + "\n" + shard_focus_marker.lstrip("\n"),
                1,
            )
            print(f"  [{shard_id}] Patch 4 applied.")
        else:
            print(f"  [{shard_id}] WARNING: 'Shard focus:' not found, Patch 4 skipped.")
    else:
        print(f"  [{shard_id}] Patch 4 already present, skipped.")

    # ── Patch 3: insert before "Banned openings for this batch:" ──
    banned_marker = "\nBanned openings for this batch:"
    if SENTINEL_3 not in content:
        if banned_marker in content:
            content = content.replace(
                banned_marker,
                "\n" + PATCH_3 + "\n" + banned_marker.lstrip("\n"),
                1,
            )
            print(f"  [{shard_id}] Patch 3 applied.")
        else:
            print(f"  [{shard_id}] WARNING: 'Banned openings' not found, Patch 3 skipped.")
    else:
        print(f"  [{shard_id}] Patch 3 already present, skipped.")

    # ── Patch 2: insert before "Quality requirements:" ──────────
    quality_marker = "\nQuality requirements:"
    if SENTINEL_2 not in content:
        if quality_marker in content:
            content = content.replace(
                quality_marker,
                "\n" + PATCH_2 + "\n" + quality_marker.lstrip("\n"),
                1,
            )
            print(f"  [{shard_id}] Patch 2 applied.")
        else:
            print(f"  [{shard_id}] WARNING: 'Quality requirements:' not found, Patch 2 skipped.")
    else:
        print(f"  [{shard_id}] Patch 2 already present, skipped.")

    return content


def main():
    shards = [f"MX_SEX_{i:03d}" for i in range(1, 13)]

    for shard_id in shards:
        path = PROMPTS_DIR / f"{shard_id}.txt"
        if not path.exists():
            print(f"[{shard_id}] File not found: {path} — skipped.")
            continue

        content = path.read_text(encoding="utf-8")
        print(f"\n[{shard_id}] Processing...")

        if shard_id == "MX_SEX_010":
            new_content = apply_patches_010(content)
        else:
            new_content = apply_patches_standard(content, shard_id)

        if new_content != content:
            path.write_text(new_content, encoding="utf-8")
            print(f"  [{shard_id}] ✓ File written.")
        else:
            print(f"  [{shard_id}] No changes (all patches already applied).")

    print("\nDone.")


if __name__ == "__main__":
    main()
