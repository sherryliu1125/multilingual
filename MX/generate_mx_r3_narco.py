#!/usr/bin/env python3
"""
MX R3 NARCO Synthetic Data Generator
Label  : MX_Narco_Culture_And_Cartel_Glorification
Model  : mistralai/mistral-small-4-119b-2603 (NVIDIA NIM)
Output : multilingual/MX/MX-R3/narco/MX_NARCO.csv

运行方式：
  python generate_mx_r3_narco.py

生成流程：
  - 并发运行 12 个 shard（001–012），每个 200 条，共 2400 条候选
  - MAX_WORKERS 个 shard 同时跑，每批完成后立即追加写入同一个 CSV（实时落盘）
  - 写入用锁保护，全局去重（跨 shard），支持断点续跑
  - 若缺口，自动补充，最多 MAX_FILL_ROUNDS 轮
"""

import os, csv, io, json, time, re, threading, concurrent.futures, urllib.request, urllib.error
from pathlib import Path

# ── .env ─────────────────────────────────────────────────────────────
def _load_dotenv(p: Path):
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

_load_dotenv(Path(__file__).resolve().parent / ".env")

# =====================================================================
# ★ CONFIG
# =====================================================================

SAMPLES_PER_SHARD: int = 200
BATCH_SIZE: int        = 55
MAX_FILL_ROUNDS: int   = 8
MAX_WORKERS: int       = 3   # 同时并发几个 shard

MODEL_NAME    = "mistralai/mistral-small-4-119b-2603"
TEMPERATURE   = 0.9
TOP_P         = 0.95
MAX_TOKENS    = 10000
MAX_RETRIES   = 3
RETRY_BACKOFF = 8

SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / "MX-R3" / "narco"
OUTPUT_FILE = OUTPUT_DIR / "MX_NARCO.csv"
CSV_COLUMNS = ["text", "final_category", "shard_id"]

# =====================================================================
# ★ PROMPT
# =====================================================================

BASE_PROMPT = """\
Eres un generador de datos sintéticos de entrenamiento para seguridad de contenido.
Tarea: generar muestras positivas de MX_Narco_Culture_And_Cartel_Glorification en español mexicano.

Definición del label:
MX_Narco_Culture_And_Cartel_Glorification identifica contenido que glorifica, romantiza,
idolatra o promueve positivamente el narco, carteles, sicarios, el estilo de vida narco
o la cultura de corridos/tumbados en contexto mexicano.
No es sobre tráfico de drogas en sí, sino sobre la glorificación cultural y la identidad cartel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Semántica que DEBE cubrirse (al menos uno por muestra):
1. Idolatría de narcos/carteles: alabar, recordar con nostalgia o idolizar a El Chapo,
   El Mencho, CJNG, Cártel de Sinaloa, Los Zetas, Los Chapitos, La Familia, etc.
   como héroes, protectores, exitosos o leyendas.
2. Narcocorrido/corridos tumbados: fragmentos líricos o comentarios de fans que
   glorifican jefe, plaza, trocas, blindadas, joyas, radios, la sierra, respeto.
3. Identidad sicario/cartel: querer ser sicario, pertenecer a la empresa, lealtad
   al jefe/patrón, sentirse parte de la maña o guardar la plaza.
4. Glorificación de la violencia narco como orden: describir el control de plaza o
   disciplina cartel como justicia, protección, respeto o poder.
5. Símbolos narco: combinación de nombre/argot cartel + emoji/simbología visual
   formando semántica glorificadora (mínimo 12 palabras, no solo símbolos).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Idioma y estilo:
* Solo español mexicano. Natural, como lo escribiría un usuario real en TikTok,
  WhatsApp, Telegram, Instagram, Twitter, foros o comentarios de YouTube.
* Puede incluir slang mexicano: compa, plebes, morro, jefe, patrón, la plaza,
  la empresa, la maña, la sierra, el rancho, troca, blindada, corridón, bélico,
  al tiro, firme, macizo, puro, arriba, ánimo.
* Puede mezclar poco inglés de redes: boss, flow, status (muy poca proporción).
* No generar texto en chino, no traducir al chino, no explicar en chino.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distribución de longitud:
* 20%: 12–24 palabras. Comentario corto, estado, fragmento de corrido.
* 70%: 25–55 palabras. Publicación social, caption, corrido completo, post de fan.
* 10%: 56–90 palabras. Caption largo, post narrativo, pequeño ensayo de fan.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROHIBIDO GENERAR (si aparece, reescribir antes de incluir en CSV):
✗ Transacciones de droga: vendo, precio, disponible, WhatsApp para comprar, entrega.
✗ Tutoriales: fabricar, esconder, rutas de tráfico, lavado de dinero, modificar armas.
✗ Amenazas reales: "voy a matar a X", levantones, cobro de piso específico.
✗ Detalles sangrientos: decapitaciones, desmembramiento, torturas, cuerpos colgados.
✗ Reclutamiento real: contacto, teléfono, lugar, sueldo, pasos para unirse.
✗ Estigmatización étnica: "los mexicanos son narcos", atacar a toda una región como grupo.
✗ Contenido safe: noticias neutrales, antinarcóticos, homenajes a víctimas,
  crítica al gobierno, comentario musical ordinario ("este corrido tiene buen ritmo").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Anti-plantilla / diversidad:
* No repetir la misma apertura de frase.
* No que más del 30% de muestras contenga "puro", "patrón", "CJNG", "Sinaloa" o "respeto".
* No que todos sean corridos; incluir posts, captions, estados, comentarios, narrativas.
* No solo "cartel + emoji". Cada muestra debe tener semántica completa entrenable.
* Diversificar: alabanza directa / identidad / defensa/normalización / glorificación de poder /
  estilo de vida lujoso / lealtad al jefe / orgullo regional narco / simbología codificada.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autocomprobación antes de cada muestra:
1. ¿Es español mexicano natural? Si no → reescribir.
2. ¿Glorifica claramente narco/cartel/sicario/corrido/estilo narco? Si no → reescribir.
3. ¿Contiene transacción, tutorial, amenaza real, detalle sangriento o reclutamiento? Si sí → descartar.
4. ¿Tiene menos de 12 palabras o es solo fragmento de entidad? Si sí → descartar.
5. ¿Es contenido safe, noticias, antinarcóticos o crítica ordinaria? Si sí → descartar.
6. ¿Es muy similar a la muestra anterior del mismo shard? Si sí → reescribir.
7. ¿Contiene etiqueta/header CSV/texto en chino/JSON/markdown? Si sí → reescribir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Formato de salida:
* CSV puro, primera línea: text,final_category
* final_category fijo: MX_Narco_Culture_And_Cartel_Glorification
* Sin explicaciones, sin numeración, sin markdown.
* Si text contiene comas o comillas, usar escape CSV estándar.
* Prohibido incluir saltos de línea dentro del campo text.
"""

SHARD_INSTRUCTIONS = {
    "001": """\
SHARD 001: Idolatría de figuras narco
Glorifica, homenajea o idolatra a El Chapo, El Mencho, CJNG, Cártel de Sinaloa,
Los Zetas, Los Chapitos, etc. como héroes, leyendas, protectores o modelos de éxito.
Incluir: tributos de fans, recuerdos nostálgicos, defensa de figura narco, lealtad a la vieja guardia.
""",
    "002": """\
SHARD 002: Identidad cartel y slogans
Gira en torno a la empresa, la maña, la plaza, el equipo, los de la sierra.
Bio de redes, firma de TikTok, slogan de comentarios, hashtag de fandom.
Puede usar emoji, pero no solo símbolos.
""",
    "003": """\
SHARD 003: Fragmentos de narcocorrido / corridos tumbados
Simula letras, rimas, captions de canciones, comentarios de fans.
Describe positivamente jefe, plaza, trocas, joyas, radios, blindadas, sierra, respeto.
No incluir detalles sangrientos ni amenazas reales.
""",
    "004": """\
SHARD 004: Glamurización del estilo de vida narco
Coches de lujo, joyas, ranchos, fiestas, poder, guardaespaldas.
El mensaje central es que el narco es éxito, estatus y meta de vida.
No mencionar transacciones concretas de droga.
""",
    "005": """\
SHARD 005: Masculinidad sicario / identidad de respeto
Ser sicario, lealtad, firmeza, disciplina, hombre de palabra.
Puede ser identidad, admiración o fanatismo.
No describir cómo matar ni amenazar a alguien.
""",
    "006": """\
SHARD 006: Glorificación del control de plaza
La plaza se respeta, aquí manda la empresa, orden, disciplina, protección.
El cartel como autoridad confiable, fuerte, justa, alternativa al Estado.
No incluir extorsión, secuestro ni ejecuciones concretas.
""",
    "007": """\
SHARD 007: Slang juvenil y expresión en redes sociales
TikTok caption, Instagram story, WhatsApp status, comentario corto, meme.
Estilo más oral, codificado, juvenil, ligero.
Debe seguir siendo semánticamente narco/cartel glorificador.
""",
    "008": """\
SHARD 008: Lealtad al jefe / patrón
Juramento de lealtad, defender al patrón, el jefe como mentor/protector/benefactor.
Puede expresar "con el jefe hasta el final".
Sin reclutamiento real, contacto, remuneración ni pasos para unirse.
""",
    "009": """\
SHARD 009: Símbolos narco y combinaciones de argot/emoji
Nombre de cartel o argot + 👑💀💸🦜🐐🔥🛻 formando semántica glorificadora.
Cada muestra mínimo 12 palabras, no solo emojis.
Combinar identidad de plaza, orgullo de cartel, símbolo de poder.
""",
    "010": """\
SHARD 010: Defensa y normalización del cartel
"No son malos, dan trabajo", "el gobierno no ayuda pero la empresa sí".
El crimen organizado como proveedor, protector, fuente de orden comunitario.
Sin operaciones ilegales concretas.
""",
    "011": """\
SHARD 011: Sabor regional / local mexicano
Sinaloa, Jalisco, Michoacán, Tamaulipas, Culiacán, la sierra, el rancho, la frontera.
El narco como identidad regional o cultural, no como estigma étnico grupal.
La semántica debe glorificar cartel/narco cultura, no atacar a los habitantes de una región.
""",
    "012": """\
SHARD 012: Casos difíciles positivos / glorificación codificada
Más sutil, literario o implícito.
Sin decir "cartel" directamente, pero usando patrón, plaza, empresa, sierra, radios,
blindadas, respeto, corridos, plebes, rancho para formar glorificación narco inequívoca.
No tan codificado que sea inentrenable.
Debe ser claramente no-safe, no comentario musical ordinario, no crítica política.
""",
}

def build_prompt(shard_id: str, n: int) -> str:
    return (
        BASE_PROMPT
        + "\n"
        + SHARD_INSTRUCTIONS[shard_id]
        + f"\nGenerar {n} muestras. Salida CSV (con encabezado text,final_category):\n"
    )

# =====================================================================
# API
# =====================================================================

def call_api(prompt: str, label: str) -> str:
    base_url = (
        os.getenv("NIM_BASE_URL") or os.getenv("NVIDIA_BASE_URL")
        or "https://integrate.api.nvidia.com/v1"
    ).rstrip("/")
    api_key = (
        os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        or os.getenv("NIW_API_KEY") or ""
    )
    if not api_key:
        raise RuntimeError("API key not found. Set NVIDIA_API_KEY.")

    for attempt in range(1, MAX_RETRIES + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": (
                    "You are a content-safety synthetic data generator. "
                    "Output only valid CSV as instructed. "
                    "No markdown, no explanation, no extra text."
                )},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        }
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                time.sleep(2)
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  [{label}] HTTP {e.code} (attempt {attempt}/{MAX_RETRIES}): {body[:400]}")
            if attempt < MAX_RETRIES:
                wait = 60 * attempt if e.code == 429 else RETRY_BACKOFF * attempt
                print(f"  [{label}] Waiting {wait}s...")
                time.sleep(wait)
        except Exception as e:
            print(f"  [{label}] Error (attempt {attempt}/{MAX_RETRIES}): {repr(e)[:300]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"[{label}] API failed after {MAX_RETRIES} attempts.")

# =====================================================================
# CSV 解析
# =====================================================================

# 常见中文字符范围检测
def _has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def _word_count(text: str) -> int:
    return len(text.split())

def parse_csv_output(raw: str, shard_id: str, label: str) -> list[dict]:
    raw = re.sub(r"```[^\n]*\n?", "", raw).strip()
    records = []
    header_skipped = False

    try:
        reader = csv.reader(io.StringIO(raw))
    except Exception:
        return records

    for row in reader:
        if not row:
            continue
        # 跳过表头
        if not header_skipped and row[0].strip().lower() in ("text", '"text"'):
            header_skipped = True
            continue

        if len(row) < 2:
            print(f"  [{label}] Skip short row: {row}")
            continue

        text = row[0].strip().strip('"')
        if not text:
            continue

        # 过滤：中文、太短、标签泄漏
        if _has_chinese(text):
            print(f"  [{label}] Chinese rejected: \"{text[:60]}\"")
            continue
        if _word_count(text) < 8:
            print(f"  [{label}] Too short: \"{text[:60]}\"")
            continue
        if "MX_Narco_Culture_And_Cartel_Glorification" in text or "final_category" in text:
            print(f"  [{label}] Label leak: \"{text[:60]}\"")
            continue

        records.append({
            "text": text,
            "final_category": "MX_Narco_Culture_And_Cartel_Glorification",
            "shard_id": shard_id,
        })

    return records

# =====================================================================
# 写入（加锁，并发安全）
# =====================================================================

_write_lock = threading.Lock()

def append_records(records: list[dict], file_exists: bool) -> bool:
    """追加写入 CSV，返回写入后 file_exists=True。"""
    with _write_lock:
        with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                w.writeheader()
            w.writerows(records)
    return True

# =====================================================================
# Shard 运行（可在线程中独立执行）
# =====================================================================

_seen_lock = threading.Lock()

def run_shard(shard_id: str, seen_texts: set[str], file_exists: bool) -> int:
    """运行单个 shard，返回本次新增条数。seen_texts 跨 shard 共享（加锁）。"""
    label = f"NARCO/{shard_id}"
    collected = 0
    round_idx = 0

    print(f"[{label}] ▶ Start (target={SAMPLES_PER_SHARD})", flush=True)

    while collected < SAMPLES_PER_SHARD and round_idx < MAX_FILL_ROUNDS:
        needed = SAMPLES_PER_SHARD - collected
        n = min(BATCH_SIZE, needed)
        print(f"[{label}/r{round_idx}] Requesting {n} rows...", flush=True)

        try:
            raw = call_api(build_prompt(shard_id, n), f"{label}/r{round_idx}")
        except Exception as e:
            print(f"[{label}/r{round_idx}] ✗ {e}", flush=True)
            round_idx += 1
            continue

        records = parse_csv_output(raw, shard_id, label)
        print(f"[{label}/r{round_idx}] Parsed {len(records)} valid rows", flush=True)

        # 去重（锁保护 seen_texts 并发读写）
        deduped = []
        with _seen_lock:
            for rec in records:
                t = rec["text"]
                if t not in seen_texts:
                    seen_texts.add(t)
                    deduped.append(rec)

        dup_removed = len(records) - len(deduped)
        if dup_removed:
            print(f"[{label}/r{round_idx}] Removed {dup_removed} dups, kept {len(deduped)}", flush=True)

        if deduped:
            remaining = SAMPLES_PER_SHARD - collected
            deduped = deduped[:remaining]
            nonlocal_fe = append_records(deduped, file_exists)
            file_exists = nonlocal_fe
            collected += len(deduped)
            print(f"[{label}/r{round_idx}] +{len(deduped)} written → {collected}/{SAMPLES_PER_SHARD}", flush=True)
        else:
            print(f"[{label}/r{round_idx}] All duplicates, retrying...", flush=True)

        round_idx += 1

    status = "✓ Complete" if collected >= SAMPLES_PER_SHARD else f"⚠ Incomplete ({collected}/{SAMPLES_PER_SHARD})"
    print(f"[{label}] {status}", flush=True)
    return collected

# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 62)
    print("  MX R3 NARCO Synthetic Data Generator")
    print("=" * 62)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Shards     : 001–012  ({SAMPLES_PER_SHARD} rows each)")
    print(f"  Total goal : {SAMPLES_PER_SHARD * 12} rows")
    print(f"  Batch size : {BATCH_SIZE} rows/call")
    print(f"  Workers    : {MAX_WORKERS} concurrent shards")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载已有数据，建立全局 seen_texts，并统计各 shard 已有行数
    seen_texts: set[str] = set()
    shard_counts: dict[str, int] = {}
    file_exists = OUTPUT_FILE.exists()

    if file_exists:
        with open(OUTPUT_FILE, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = row.get("text", "").strip()
                if t:
                    seen_texts.add(t)
                sid = row.get("shard_id", "")
                shard_counts[sid] = shard_counts.get(sid, 0) + 1
        print(f"\n  Resumed: {len(seen_texts)} existing rows loaded")
        for sid, cnt in sorted(shard_counts.items()):
            print(f"    shard {sid}: {cnt} 行")

    all_shards = [f"{i:03d}" for i in range(1, 13)]
    shards_to_run = []
    for sid in all_shards:
        cnt = shard_counts.get(sid, 0)
        if cnt >= SAMPLES_PER_SHARD:
            print(f"[NARCO/{sid}] Already complete ({cnt} rows), skipped.")
        else:
            shards_to_run.append(sid)

    if not shards_to_run:
        print("\nAll shards complete!")
        return

    print(f"\n  待跑 shards: {shards_to_run}\n")

    total_added = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_shard, sid, seen_texts, file_exists): sid
            for sid in shards_to_run
        }
        for fut in concurrent.futures.as_completed(futures):
            sid = futures[fut]
            try:
                added = fut.result()
                total_added += added
            except Exception as e:
                print(f"[NARCO/{sid}] ✗ FATAL: {e}", flush=True)

    print(f"\n{'=' * 62}")
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8", newline="") as f:
            final_total = sum(1 for _ in csv.DictReader(f))
        print(f"  文件总行数：{final_total} 行")
    print(f"  本次新增：{total_added} 行")
    print("=" * 62)


if __name__ == "__main__":
    main()
