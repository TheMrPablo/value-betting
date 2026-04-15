# Value Betting — Contexto para nueva sesión con Claude

Pegá este archivo al inicio de una nueva conversación para que Claude tenga contexto completo del proyecto.

---

## Qué es este proyecto

Sistema automatizado de value betting en fútbol europeo (Bundesliga 2, con Eredivisie planeada). Detecta ineficiencias de mercado usando modelos estadísticos y envía alertas por Telegram para que el usuario decida si apuesta o no.

**Repo**: https://github.com/TheMrPablo/value-betting  
**Stack**: Python 3.11 · SQLite · GitHub Actions · Telegram Bot · Claude API (opcional)  
**Bankroll actual**: 500,000 COP · Período de burn-in: primeras 50 apuestas a 1/8 Kelly

---

## Arquitectura en una línea

```
CSVs football-data.co.uk
  → load_historical.py → fixtures DB
  → feature_builder.py → match_features
  → dixon_coles.py + xgboost_model.py → predictions
  → value_engine.py → signals (filtros + Kelly)
  → [betting_agent.py] → análisis Claude (opcional)
  → telegram_bot.py → alerta al usuario (GO / SKIP / HALF)
```

---

## Archivos clave

| Archivo | Qué hace |
|---------|----------|
| `src/models/dixon_coles.py` | Modelo Poisson bivariado con corrección tau. Predice Over/Under, 1X2, BTTS |
| `src/models/xgboost_model.py` | XGBoost calibrado (Platt scaling). 12 features de forma, H2H, descanso, posición |
| `src/features/feature_builder.py` | Construye las features desde la DB y las guarda en `match_features` |
| `src/engine/value_engine.py` | Motor central: ensemble DC+XGB, edge, Kelly, filtros → genera señales |
| `src/engine/kelly.py` | Kelly fraccionario (1/4 conservador, 1/8 agresivo/burn-in) con caps |
| `src/engine/filters.py` | 15 filtros pre-apuesta: overround, temporada, clima, modelo, exposición... |
| `src/ingestion/odds_scraper.py` | Scraper de Kambi/Betplay sin auth. Mapeo de nombres de equipos |
| `src/output/betting_agent.py` | Llama a Claude para analizar señales (busca noticias, lesiones, decide) |
| `src/output/telegram_bot.py` | Envía alertas al usuario. Procesa callbacks GO/SKIP/HALF |
| `scripts/init_db.py` | Crea el schema SQLite (12 tablas) |
| `scripts/load_historical.py` | Descarga y carga CSVs de football-data.co.uk |
| `config/thresholds.yaml` | Perfiles conservador/agresivo, pesos del ensemble (60% DC / 40% XGB) |
| `config/bankroll.yaml` | Bankroll actual en COP, objetivo mensual, stop-loss |
| `config/leagues.yaml` | Ligas activas con códigos de API |

---

## Base de datos SQLite (`data/betting.db`)

Tablas principales:
- `fixtures` — todos los partidos (históricos + próximos), `status = finished | scheduled`
- `odds` — cuotas por fixture/mercado, `bookmaker = betplay | bet365_historical`
- `match_features` — features XGBoost pre-calculadas (forma, H2H, descanso, posición)
- `predictions` — salidas de modelos; `model = dixon_coles | xgboost`; `UNIQUE(fixture_id, model, market, selection)`
- `signals` — value bets generadas; `UNIQUE(fixture_id, market, selection)`; `telegram_sent: 0=pendiente, 1=enviada, 2=descartada`
- `bets` — decisiones del usuario (GO/SKIP/HALF con stake real)
- `results` — resultados liquidados con P&L en COP
- `performance_snapshots` — snapshots semanales de ROI/Brier/bankroll

**La DB persiste como GitHub Actions artifact** (retención 7 días en daily pipeline, 1 día en prematch scan). Si el artifact expira, la DB se recrea desde cero con los CSVs históricos.

---

## Workflows (GitHub Actions)

| Workflow | Cuándo | Qué hace |
|----------|--------|----------|
| `daily_pipeline.yml` | 6am COT (11 UTC) | Retrain completo + odds + señales + [agente] + Telegram |
| `prematch_scan.yml` | Cada 30min | Solo si hay partidos en 8h: odds frescos + señales + [agente] + Telegram + listen callbacks |
| `weekly_report.yml` | Lunes 8am COT | Resumen semanal + revisión del agente |

**Variable de repo `USE_AGENT`** (GitHub Settings → Variables → Actions):
- `true` (default) → el agente Claude analiza cada señal antes de enviarla
- `false` → señales van directo a Telegram sin filtro del agente

---

## Configuración del agente Claude

El agente vive en `betting_agent.py`. Hace dos llamadas por señal:
1. Busca noticias recientes (web search) sobre los equipos
2. Analiza la señal y decide si recomendar o descartar

**Caché dinámico** según cercanía al partido:
- \>24h antes → caché 6h
- 6-24h → caché 2h
- 2-6h → caché 1h
- <2h → sin caché (alineaciones oficiales)

**Descarte automático sin gastar API**: edge >25% (modelo desactualizado).

**Secreto requerido**: `ANTHROPIC_API_KEY` en GitHub Secrets.

---

## Fixes aplicados en la sesión de abril 2026

### Problema original
El agente gastó $10 USD en créditos porque había señales duplicadas (36 en vez de 9), lo que generaba 72 llamadas a la API por corrida.

### Causa raíz y fixes aplicados

**1. Señales duplicadas** (`src/engine/value_engine.py` — `_save_signals`)
- Antes: `INSERT OR IGNORE` sin UNIQUE constraint → cada corrida del pipeline insertaba filas nuevas
- Fix: smart upsert — si ya fue enviada/descartada no la toca; si está pendiente la reemplaza; si no existe la inserta

**2. Análisis del agente no aparecía en Telegram** (`src/output/telegram_bot.py`)
- Antes: `betting_agent.py` sobreescribía `signals.flags` (JSON array) con un dict. `telegram_bot.py` iteraba el dict como lista → crash silencioso
- Fix: `send_pending_signals` detecta si `flags` es dict o lista y extrae `flags_detail` + `agent_analysis` correctamente

**3. Bankroll incorrecto** (`src/engine/value_engine.py` — `load_bankroll`)
- Antes: fallback hardcodeado a 150K COP cuando no había snapshots
- Fix: lee `config/bankroll.yaml` (500K) antes del hardcode

**4. Modelo desactualizado** (`src/output/betting_agent.py`)
- `claude-sonnet-4-5` → `claude-sonnet-4-6`

**5. UNIQUE constraints** (`scripts/init_db.py`)
- `signals(fixture_id, market, selection)`
- `predictions(fixture_id, model, market, selection)`
- Solo aplica a DBs nuevas; DBs existentes usan el smart upsert

---

## Estado actual (abril 2026)

- ✅ Burn-in activo (primeras 50 apuestas a 1/8 Kelly)
- ✅ BL2 (Bundesliga 2) operativa
- ✅ Pipeline corrió exitosamente en modo SIN agente (`USE_AGENT=false`)
- ✅ `USE_AGENT=false` seteado como variable de repo (hasta confirmar que no hay más duplicados)
- 🔲 NL1 (Eredivisie) — configurada en `leagues.yaml` pero el pipeline solo corre BL2
- 🔲 Lesiones — campos en `match_features` pero sin fuente de datos (Fase 2)
- 🔲 Clima — tabla `weather` creada pero sin integración (Fase 2, Yr.no API)
- 🔲 Incentivos asimétricos — filtro placeholder en `filters.py` (Fase 2)
- 🔲 `performance_snapshots` — tabla creada pero sin código que la poblee automáticamente
- 🔲 Bankroll snapshot semanal — necesita un step en `weekly_report.yml`

---

## Cosas que faltan / próximos pasos sugeridos

1. **Poblar `performance_snapshots` automáticamente** — agregar step en `weekly_report.yml` que calcule ROI, Brier score y bankroll actual y lo inserte en la tabla. Ahora `load_bankroll()` siempre lee el yaml porque nunca hay snapshots.

2. **Reactivar el agente** — cuando estés seguro de que no hay más duplicados, cambiar `USE_AGENT=true` en GitHub Variables.

3. **Activar NL1** — el workflow solo corre BL2. Para agregar Eredivisie: duplicar los steps con `--league NL1` o parametrizar el loop en el workflow.

4. **Resolver warning de Node.js 20** en los workflows (minoría, no afecta funcionalidad hasta sep 2026).

5. **Test de los callbacks de Telegram** — verificar que GO/SKIP/HALF se registran correctamente en la tabla `bets`.

---

## Cómo correr localmente

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Inicializar DB (solo una vez)
python scripts/init_db.py

# Cargar históricos
python scripts/load_historical.py

# Correr modelos
python src/features/feature_builder.py --league BL2
python src/models/dixon_coles.py --league BL2 --seasons 2022-23 2023-24 2024-25 2025-26
python src/models/xgboost_model.py --league BL2

# Ver señales
python src/engine/value_engine.py --league BL2 --preview

# Enviar a Telegram (sin agente)
python src/output/telegram_bot.py --send

# Con agente
python src/output/betting_agent.py --analyze
python src/output/telegram_bot.py --send
```

**Variables de entorno** (`.env` o export):
```
ODDS_API_KEY=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
ANTHROPIC_API_KEY=...   # solo si USE_AGENT=true
```
