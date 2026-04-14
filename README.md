# Sistema Personal de Value Betting — Documentación Completa

**Repo:** `TheMrPablo/value-betting` (privado)  
**Última actualización:** 14 de abril 2026  
**Estado:** Sistema completo en producción — período de burn-in activo

---

## Índice

1. [Qué es este proyecto](#1-qué-es-este-proyecto)
2. [Filosofía y expectativas realistas](#2-filosofía-y-expectativas-realistas)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Stack técnico](#4-stack-técnico)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Base de datos](#6-base-de-datos)
7. [Modelos estadísticos](#7-modelos-estadísticos)
8. [Pipeline de datos](#8-pipeline-de-datos)
9. [Motor de decisiones](#9-motor-de-decisiones)
10. [Agente Claude](#10-agente-claude)
11. [Bot de Telegram](#11-bot-de-telegram)
12. [GitHub Actions](#12-github-actions)
13. [Ligas y mercados](#13-ligas-y-mercados)
14. [Gestión del bankroll](#14-gestión-del-bankroll)
15. [Filtros y reglas de exclusión](#15-filtros-y-reglas-de-exclusión)
16. [Resultados del backtest](#16-resultados-del-backtest)
17. [Operativa diaria](#17-operativa-diaria)
18. [Configuración y secretos](#18-configuración-y-secretos)
19. [Bugs conocidos y fixes pendientes](#19-bugs-conocidos-y-fixes-pendientes)
20. [Roadmap Fase 2](#20-roadmap-fase-2)
21. [Decisiones de diseño](#21-decisiones-de-diseño)

---

## 1. Qué es este proyecto

Sistema automatizado de value betting en fútbol europeo (Bundesliga 2). El sistema:

1. Descarga datos históricos de partidos y cuotas
2. Entrena modelos estadísticos (Dixon-Coles + XGBoost) para estimar probabilidades de Over/Under 2.5
3. Scrapa odds en tiempo real de Betplay (Colombia) via API de Kambi
4. Calcula si hay "value" — cuando el modelo estima mayor probabilidad que lo que paga Betplay
5. Un agente Claude busca noticias recientes (lesiones, alineaciones) y decide si recomendar la apuesta
6. Si aprueba, manda una alerta al celular via Telegram con los botones "Aposté / Pasé / Aposté la mitad"
7. El usuario coloca la apuesta manualmente en Betplay y confirma en el bot

**El usuario no hace nada más que colocar la apuesta cuando llega la alerta.**

---

## 2. Filosofía y expectativas realistas

### Lo que dice la literatura académica

- Los márgenes (overround) son mayores en ligas menores — confirmado (Winkelmann et al. 2023)
- Solo 3 de 11 mercados europeos muestran ineficiencias explotables con max-odds cross-bookmaker (Angelini & De Angelis 2019)
- Las ineficiencias son de corta duración y no persisten sistemáticamente (Winkelmann et al. 2024)

### Dónde viene el edge real

No de tener modelos mejores que el bookie. Viene de:
1. **Information edge local**: lesiones anunciadas antes de que el mercado las procese
2. **Mercados secundarios** (O/U, BTTS) donde los bookies modelan menos
3. **Aprovechar la opening line** antes de que los sharps la muevan

### Expectativas honestas

Con bankroll de 150,000 COP (~USD 38), solo Betplay, y BL2 O/U como mercado:
- Esto es más un **ejercicio de aprendizaje + posible edge marginal** que una máquina de dinero
- El ROI del backtest (+10.24% ensemble) es histórico — en producción será menor
- El burn-in (primeras 50 apuestas) es obligatorio antes de escalar

---

## 3. Arquitectura del sistema

```
football-data.co.uk CSVs (histórico)
        ↓
load_historical.py → betting.db
        ↓
feature_builder.py → match_features
        ↓
dixon_coles.py ──┐
                 ├→ predictions → value_engine.py
xgboost_model.py ┘                      ↓
                              (ensemble + Kelly + filtros)
                                         ↓
odds_scraper.py (Kambi/Betplay API) ────→ signals
                                         ↓
                              betting_agent.py (Claude)
                              busca noticias → decide
                                         ↓
                              telegram_bot.py → alerta
                                         ↓
                              Usuario coloca apuesta en Betplay
                                         ↓
                              Bot registra decisión → bets
```

### Flujo temporal

```
T-72h   cron diario: descarga datos, entrena modelos
T-48h   scraper empieza a buscar odds para jornada próxima
T-24h   value engine corre, genera señales candidatas
T-6h    agente Claude analiza señales con noticias recientes
T-2h    señales aprobadas enviadas a Telegram
T-0h    usuario coloca apuesta en Betplay, confirma en bot
```

---

## 4. Stack técnico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Base de datos | SQLite (archivo único, sin servidor) |
| Modelo base | Dixon-Coles 1997 (Poisson bivariado) |
| Modelo secundario | XGBoost calibrado (Platt scaling) |
| Odds scraping | API REST de Kambi (sin auth, sin Playwright) |
| Datos históricos | football-data.co.uk CSVs |
| Agente IA | Claude Sonnet (Anthropic API) |
| Alertas | Telegram Bot API |
| Automatización | GitHub Actions |
| OS desarrollo | Windows (PowerShell) |
| OS producción | Ubuntu (GitHub Actions runners) |

### Dependencias principales

```
httpx          # HTTP async
pandas         # manipulación de datos
scipy          # optimización (Dixon-Coles)
xgboost        # modelo ML
scikit-learn   # calibración Platt
feedparser     # RSS feeds
```

---

## 5. Estructura del repositorio

```
value-betting/
├── .github/
│   └── workflows/
│       ├── daily_pipeline.yml      # cron 6am COT — pipeline completo
│       ├── prematch_scan.yml       # cada 30min — odds + señales
│       └── weekly_report.yml       # lunes 8am COT — resumen
│
├── config/
│   ├── leagues.yaml                # ligas activas y configuración
│   ├── thresholds.yaml             # edge mínimo, Kelly, overround máx
│   ├── telegram.json               # ← GITIGNORED — token + chat_id
│   └── agent.json                  # ← GITIGNORED — Anthropic API key
│
├── data/
│   ├── historical/                 # CSVs de football-data.co.uk
│   │   ├── D2_2122.csv
│   │   ├── D2_2223.csv
│   │   ├── D2_2324.csv
│   │   ├── D2_2425.csv
│   │   └── D2_2526.csv
│   └── betting.db                  # ← GITIGNORED — base de datos SQLite
│
├── models/
│   ├── dc_params_BL2.pkl           # ← GITIGNORED — parámetros DC
│   └── xgb_calibrated_BL2.pkl     # ← GITIGNORED — modelo XGB
│
├── scripts/
│   ├── init_db.py                  # crea schema SQLite desde cero
│   ├── load_historical.py          # carga CSVs históricos a DB
│   └── insert_fixtures.py          # inserta fixtures futuros manualmente
│
├── src/
│   ├── ingestion/
│   │   ├── data_fetcher.py         # descarga CSVs y fixtures
│   │   └── odds_scraper.py         # API Kambi/Betplay → odds
│   │
│   ├── features/
│   │   └── feature_builder.py      # construye features para XGBoost
│   │
│   ├── models/
│   │   ├── dixon_coles.py          # modelo Poisson bivariado
│   │   └── xgboost_model.py        # modelo ML calibrado
│   │
│   ├── engine/
│   │   ├── value_engine.py         # motor central — ensemble + Kelly
│   │   ├── kelly.py                # calculadora Kelly fraccionario
│   │   └── filters.py              # reglas de exclusión §5
│   │
│   └── output/
│       ├── telegram_bot.py         # bot con botones inline
│       └── betting_agent.py        # agente Claude — análisis + decisión
│
├── requirements.txt
├── .gitignore
└── README.md (este archivo)
```

---

## 6. Base de datos

SQLite, archivo único `data/betting.db`. Se recrea desde cero en cada corrida de GitHub Actions usando los CSVs descargados.

### Tablas principales

| Tabla | Descripción |
|---|---|
| `fixtures` | Partidos históricos y futuros |
| `odds` | Cuotas por fixture/bookmaker/mercado |
| `predictions` | Probabilidades de DC y XGB |
| `match_features` | Features contextuales para XGB |
| `signals` | Señales generadas por el value engine |
| `bets` | Decisiones del usuario (GO/SKIP) |
| `results` | Outcomes y P&L |
| `performance_snapshots` | Métricas semanales |

### Campos clave en signals

```sql
telegram_sent:
  0 = pendiente de enviar
  1 = enviada al bot
  2 = descartada por el agente Claude
```

### Nombres de equipos BL2 en la DB

Los nombres vienen del CSV de football-data.co.uk:
```
Bielefeld, Bochum, Braunschweig, Darmstadt, Dresden, Elversberg,
Fortuna Dusseldorf, Greuther Furth, Hannover, Hertha, Holstein Kiel,
Kaiserslautern, Karlsruhe, Magdeburg, Nurnberg, Paderborn,
PreuÃen MÃ¼nster (encoding roto — así está en la DB),
Schalke 04
```

**Nota:** "PreuÃen MÃ¼nster" es el encoding roto de "Preußen Münster". El sistema lo maneja así internamente; no rompe nada.

---

## 7. Modelos estadísticos

### Dixon-Coles (modelo base, 60% del ensemble)

Implementa el paper original de Dixon & Coles (1997). Estima para cada equipo:
- `alpha` = fuerza atacante
- `beta` = solidez defensiva
- `gamma` = ventaja local
- `rho` = corrección de correlación para resultados bajos (0-0, 1-0, 0-1, 1-1)

La tasa esperada de goles es:
```
lambda_home = alpha_home × beta_away × gamma
lambda_away = alpha_away × beta_home
```

Usa **decaimiento temporal** (xi=0.0018) — los partidos recientes pesan más.

**Resultados BL2 (entrenado con 5,112 partidos, temporadas 2021-22 a 2025-26):**
```
gamma = 1.212  (ventaja local ~21%)
rho   = -0.0687
NLL   = 4836.87
```

Equipos más ofensivos: Hamburg (1.70), Heidenheim (1.54), Holstein Kiel (1.45)  
Equipos más defensivos (débiles ofensivamente): Regensburg (0.63), Hansa Rostock (0.70)

### XGBoost calibrado (modelo secundario, 40% del ensemble)

Features usadas (12):
- Forma últimos 5 partidos: goles marcados/recibidos, tasa O/U 2.5
- H2H: tasa histórica O/U entre ambos equipos
- Descanso: días desde último partido (local y visitante)
- Posición en tabla (local y visitante)

Calibración: Platt scaling (CalibratedClassifierCV de scikit-learn)

**Métricas:**
```
Brier score global: 0.2378 (vs DC: 0.2405)
ROC-AUC: 0.6341 (random = 0.50)
```

### Ensemble

```python
prob_ensemble = 0.60 * prob_dc + 0.40 * prob_xgb
```

Pesos configurables en `config/thresholds.yaml`.

---

## 8. Pipeline de datos

### Fuente de datos históricos

**football-data.co.uk** — CSVs descargados directamente.

- Bundesliga 2: `D2.csv` (o `D2_SSYY.csv` por temporada)
- Columnas usadas: `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `B365>2.5`, `B365<2.5`
- **Bloquea IPs colombianas** — se descarga desde GitHub Actions (IPs americanas/europeas no bloqueadas)

### Fuente de odds en tiempo real

**API REST de Kambi** (plataforma de Betplay) — sin autenticación, sin Playwright.

```
Listado de eventos:
GET https://us1.offering-api.kambicdn.com/offering/v2018/betplay/listView/football/germany.json
  ?lang=es_CO&market=CO&client_id=200&channel_id=1&includeParticipants=true

Cuotas por evento:
GET https://us1.offering-api.kambicdn.com/offering/v2018/betplay/betoffer/event/{EVENT_ID}.json
  ?lang=es_CO&market=CO&client_id=200&channel_id=1&includeParticipants=true
```

Filtro O/U 2.5: `criterion.label == "Total de goles"` y `line == 2500`.
Las cuotas vienen en milésimas: `odds / 1000 = odds_decimal`.

**Overround real de Betplay en BL2:** ~9-11% (umbral ajustado a 11% en thresholds.yaml).

### Fixtures futuros

Los CSVs de football-data.co.uk solo tienen partidos jugados. Los fixtures futuros se insertan via `scripts/insert_fixtures.py` con las jornadas del calendario. En producción, esto se puede automatizar con API-Football (100 requests/día gratis).

---

## 9. Motor de decisiones

### value_engine.py

1. Carga predicciones DC y XGB desde la DB
2. Calcula probabilidad ensemble (60/40)
3. Compara contra implied probability de Betplay
4. Calcula edge = prob_ensemble - implied_prob
5. Corre todos los filtros (ver §15)
6. Si pasa filtros y edge > umbral → calcula stake con Kelly
7. Genera objeto `BettingSignal` y guarda en tabla `signals`

### kelly.py

```python
kelly_fraction = edge / (odds - 1)
stake_raw = bankroll × kelly_fraction × kelly_multiplier

# Caps duros:
stake_final = min(stake_raw, bankroll × 0.05)  # nunca más de 5% del bankroll
# Y verificar que exposure total < 15% del bankroll
```

| Perfil | Edge mínimo | Fracción Kelly | Stake típico (150k) |
|---|---|---|---|
| burn_in | ≥ 5% | 1/8 | ~3,000 COP |
| conservador | ≥ 5% | 1/4 | ~6,000 COP |
| agresivo | ≥ 8% | 1/8 | ~3,000 COP |

**Burn-in**: primeras 50 apuestas reales usan 1/8 Kelly para validar calibración antes de escalar.

---

## 10. Agente Claude

### Rol

Antes de enviar cualquier señal al usuario, el agente:
1. **Descarte rápido**: si edge > 25% → SKIP sin llamar a la API (ruido del modelo)
2. **Busca noticias**: web search con Claude sobre lesiones y alineaciones del partido
3. **Analiza**: evalúa si el edge es creíble dado el contexto
4. **Decide**: RECOMENDAR o DESCARTAR con razón en lenguaje simple

### Configuración

- Modelo: `claude-sonnet-4-5`
- API key: guardada en `config/agent.json` y GitHub Secret `ANTHROPIC_API_KEY`
- Costo estimado: ~$0.35 USD/mes en producción normal (~4-8 señales/semana)
- Rate limit: tier gratuito tiene límites bajos — añadir $5-10 de crédito en console.anthropic.com para eliminar 429s

### Criterios de decisión del agente

**DESCARTAR si:**
- Edge > 25% (casi siempre ruido del modelo, no oportunidad real)
- Divergencia modelo/mercado > 20 puntos sin razón clara
- Noticias muestran 3+ titulares fuera por lesión
- Primera o última jornada de la temporada

**RECOMENDAR si:**
- Edge entre 5% y 20%
- Sin noticias alarmantes
- Modelo y mercado relativamente alineados (diferencia < 15 puntos)

### Revisión semanal

Cada lunes el agente revisa las señales descartadas de la semana y analiza cuántas habrían ganado o perdido, generando recomendaciones para ajustar los criterios de filtrado.

### Estado `telegram_sent` en signals

```
0 = pendiente (aprobada por agente, lista para enviar)
1 = enviada al bot
2 = descartada por agente (no se envía)
```

---

## 11. Bot de Telegram

### Configuración

- Token: en `config/telegram.json` (nunca en el código)
- Chat ID: 1414383998
- **Rotar el token** cada vez que se comparta en una conversación pública

### Formato del mensaje

```
🎯 Bundesliga 2
Holstein Kiel vs Kaiserslautern
📅 Viernes 17 Abr · 11:30am (COT)

👉 apostá a que el partido termina con
   3 o más goles (Over 2.5)
   Cuota Betplay: 1.60

💰 Apostá: $3,000 COP
   Si ganás: +$1,800 COP
   Si perdés: -$3,000 COP

━━━━━━━━━━━━━━━━━━━━
📊 El modelo estima 68% de probabilidad.
Betplay paga como si fuera 63%.
Ventaja estimada: +6%

⚠️ Período de prueba — apuesta 2/50.

🤖 Análisis: Edge moderado del 6% con modelos
alineados, baja defensiva en Kiel que favorece
partidos abiertos.

[✅ Aposté $3,000] [❌ Pasé] [½ Aposté $1,500]
```

### Flujo de respuesta

1. Usuario toca **✅ Aposté** → el bot registra en tabla `bets` con `user_decision='GO'`
2. Usuario toca **❌ Pasé** → registra `user_decision='SKIP'`
3. Usuario toca **½ Aposté $X** → registra GO con stake reducido a la mitad
4. Los botones desaparecen después de tocar uno (evita doble tap)
5. Se envía confirmación: "Apostá $3,000 COP @ 1.60 en Betplay"

### Comandos CLI

```powershell
python src/output/telegram_bot.py --test    # mensaje de prueba
python src/output/telegram_bot.py --send    # enviar señales pendientes
python src/output/telegram_bot.py --listen  # procesar respuestas del usuario
python src/output/telegram_bot.py --weekly  # enviar resumen semanal
```

---

## 12. GitHub Actions

### Workflows

**daily_pipeline.yml** — cron `0 11 * * *` (6am COT):
1. Descarga CSVs históricos de football-data.co.uk (IPs de GitHub no bloqueadas)
2. Inicializa DB si no existe
3. Carga datos a DB (`load_historical.py`)
4. Construye features (`feature_builder.py`)
5. Entrena Dixon-Coles (~22 min en runners de GitHub)
6. Entrena XGBoost (~30 seg)
7. Scrapa odds de Betplay (Kambi API)
8. Corre value engine
9. Agente analiza señales
10. Bot envía señales aprobadas a Telegram
11. Sube DB como artifact (retention 7 días)
12. Si falla algún paso → notificación en Telegram

**prematch_scan.yml** — cron `0,30 * * * *` (cada 30 min):
- Solo corre si hay fixtures en las próximas 8 horas
- Actualiza odds + señales + escucha respuestas del bot

**weekly_report.yml** — cron `0 13 * * 1` (lunes 8am COT):
- Envía resumen semanal de P&L y calibración
- Agente genera análisis de señales descartadas

### GitHub Secrets configurados

| Secret | Uso |
|---|---|
| `TELEGRAM_TOKEN` | Token del bot de Telegram |
| `TELEGRAM_CHAT_ID` | ID del chat personal (1414383998) |
| `ANTHROPIC_API_KEY` | API key de Claude |

### Nota sobre minutos de Actions

Free tier: 2,000 min/mes. El daily pipeline tarda ~30 min. Con corridas 5 días/semana = ~600 min/mes. Holgado.

---

## 13. Ligas y mercados

### Fase 1 (activa)

| Liga | Estado | Mercado | Overround Betplay |
|---|---|---|---|
| **Bundesliga 2** | ✅ Activa | Over/Under 2.5 | ~9-11% |
| Eredivisie | ❌ Excluida Fase 1 | — | — |

**Eredivisie excluida**: backtest dio ROI -8.59%. Se retoma en Fase 2 con más features contextuales.

### Por qué Bundesliga 2

- Datos históricos completos desde 1993/94 (football-data.co.uk)
- Integridad alta (DFB, sin escándalos de match-fixing)
- Betplay la lista consistentemente con mercados O/U
- Media de goles estable (~2.8/partido)
- Overround razonable para Colombia (~10%)

### Por qué Over/Under 2.5

Dixon-Coles es un modelo de goles — su ventaja natural está en O/U, no en 1X2. El edge en 1X2 requiere competir directamente contra los modelos del bookie, que son muy buenos.

### Ligas excluidas explícitamente

- **Belgian Pro League**: match-fixing sistémico (Operation Propere Handen 2017-2019, 57 imputados, €121M en impuestos evadidos)
- **EFL Championship**: volumen muy alto, mercado muy eficiente

---

## 14. Gestión del bankroll

### Bankroll inicial

150,000 COP (~USD 38). Reinversión iterativa.

### Sizing

```
Durante burn-in (primeras 50 apuestas):
  Kelly multiplier = 1/8
  Stake típico = 3,000 COP (2% bankroll)

Después del burn-in (si ROI ≥ -2%):
  Perfil conservador: Kelly 1/4, stake ~6,000 COP
  Perfil agresivo: Kelly 1/8, stake ~3,000 COP

Caps duros (siempre):
  Máximo 5% del bankroll por apuesta
  Máximo 15% del bankroll expuesto simultáneamente
```

### Hitos

| Hito | Condición | Acción |
|---|---|---|
| Hito 1 | ROI > 0% en backtest | ✅ Cumplido (+10.24% ensemble) |
| Hito 2 | 10 paper trades calibrados | ⏳ En progreso |
| Hito 3 | 50 apuestas reales, ROI ≥ -2% | ⏳ Pendiente |

### Reglas de actualización

- Recalcular unidad base cada lunes según bankroll actual
- Si el bankroll duplica → retirar 20% (primer corte: al llegar a 300k, retirar 60k)
- Stop-loss mensual: si perdés > 25% en un mes → pausar 7 días y auditar el modelo

---

## 15. Filtros y reglas de exclusión

### Filtros duros (→ SKIP automático)

| Código | Condición |
|---|---|
| `HIGH_OVERROUND` | Overround > 11% en el mercado |
| `LATE_MARKET` | Mercado abierto < 3h antes del kickoff |
| `EARLY_SEASON` | Primeras 2 jornadas de la temporada |
| `END_SEASON` | Últimas 2 jornadas (incentivos distorsionados) |
| `POST_BREAK` | Primera jornada post-parón internacional |
| `EXPOSURE_CAP` | Exposición total ≥ 15% del bankroll |

### Filtros suaves (→ REVIEW, requiere confirmación del agente)

| Código | Condición |
|---|---|
| `BURN_IN` | Primeras 50 apuestas (stake reducido) |
| `COACH_CHANGE` | Cambio de entrenador < 3 partidos |
| `WEATHER_SNOW` | Nieve activa |
| `WEATHER_WIND` | Viento > 30 km/h |
| `WEATHER_RAIN` | Lluvia intensa |
| `DERBY` | Derbi regional con tensión emocional |
| `MODEL_DISAGREE` | DC y XGB divergen > 10 puntos porcentuales |
| `LINE_MOVE` | Movimiento de línea > 15% sin noticia pública |

### Regla del agente sobre edges

- Edge > 25% → SKIP directo sin llamar a la API
- Edge 5-25% → el agente analiza con noticias antes de decidir

---

## 16. Resultados del backtest

### Dixon-Coles solo — BL2

| Temporadas test | Apuestas | ROI | Brier score |
|---|---|---|---|
| 2023-24 + 2024-25 | 227 | +5.12% | 0.2405 |
| 2025-26 (parcial) | 100 | +1.84% | 0.2515 |
| 3 temporadas | 1,028 | +4.40% | 0.2429 |

### Ensemble DC+XGB — BL2

| Temporadas test | Apuestas | ROI | Brier score |
|---|---|---|---|
| **3 temporadas** | **535** | **+10.24%** | **0.2464** |

### Over 2.5 rate real

- BL2 2025-26: **58.2%** (consistente con temporadas anteriores ~59.7%)
- El modelo está bien calibrado en términos de tasa general

### Nota importante sobre el backtest

El +10.24% usa odds de cierre históricas de bet365. En producción con Betplay:
- Overround es ~10% vs ~5% de bet365 → el ROI real será menor
- Los edges de backtest no se traducen 1:1 a producción

---

## 17. Operativa diaria

### Lo que hace el sistema solo

```
Lunes 6am COT:
  → Descarga D2_2526.csv actualizado
  → Recarga datos a DB
  → Reentrena DC y XGBoost
  → Genera señales para la semana

Días con partidos (viernes, sábado, domingo):
  → Cada 30 min actualiza odds de Betplay
  → Si hay value → agente analiza → si aprueba → alerta en Telegram

Lunes 8am COT:
  → Resumen semanal: P&L, ROI, calibración
  → Análisis del agente sobre señales descartadas
```

### Lo que hace el usuario

```
Cuando llega alerta en Telegram (2-5 veces por semana en temporada):
  1. Leer el mensaje (30 segundos)
  2. Si querés apostar → abrí Betplay, buscá el partido, apostá el monto indicado
  3. Tocar "✅ Aposté" en el bot
  4. Listo — el sistema registra todo automáticamente
```

### Lo que NO hay que hacer

- No correr scripts manualmente (el pipeline lo hace solo)
- No actualizar CSVs manualmente (GitHub Actions los descarga)
- No calcular el stake manualmente (Kelly lo hace solo)
- No llevar un Excel de seguimiento (la DB lo registra todo)

---

## 18. Configuración y secretos

### Archivos de configuración (en .gitignore — NUNCA subir al repo)

**config/telegram.json**
```json
{
  "token": "TU_TOKEN_DEL_BOT",
  "chat_id": "1414383998"
}
```

**config/agent.json**
```json
{
  "api_key": "TU_ANTHROPIC_API_KEY"
}
```

### GitHub Secrets

Configurados en `Settings → Secrets and variables → Actions`:
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`
- `ANTHROPIC_API_KEY`

### Rotar el token de Telegram

Hacerlo cada vez que el token se comparta en una conversación:
1. Abrir Telegram → buscar `@BotFather`
2. `/mybot` → seleccionar el bot → `API Token` → `Revoke current token`
3. Actualizar `config/telegram.json` y el GitHub Secret `TELEGRAM_TOKEN`

---

## 19. Bugs conocidos y fixes pendientes

### Aplicados

- ✅ Logging format `%,` no soportado en Python → cambiado a `%.0f`
- ✅ `heavy_rain/high_wind/snow` no están en schema de Fase 1 → columnas removidas del query
- ✅ Token de Telegram expuesto en conversación → rotado
- ✅ Fallback del agente era permisivo → ahora siempre SKIP si falla la API
- ✅ Retry con delay en 429 de Anthropic API

### Pendientes

**Bug crítico — Señales duplicadas:**
En `src/engine/value_engine.py`, cambiar:
```python
INSERT OR REPLACE INTO signals  # malo — genera duplicados
```
por:
```python
INSERT OR IGNORE INTO signals   # correcto
```

**Rate limit API de Anthropic:**
Añadir $5-10 de crédito en console.anthropic.com para eliminar 429s.
Mientras tanto, el `time.sleep(5)` entre señales reduce el problema.

**Preußen Münster encoding:**
El nombre aparece como `PreuÃen MÃ¼nster` en la DB (encoding roto del CSV).
El modelo DC muestra warnings "equipo no visto en entrenamiento — usando medias".
Fix: normalizar el nombre al cargar el CSV en `load_historical.py`.

**Fixtures futuros automáticos:**
Actualmente se insertan manualmente con `insert_fixtures.py`.
Fix en Fase 2: usar API-Football (100 req/día gratis) para obtener fixtures automáticamente.

---

## 20. Roadmap Fase 2

### Mejoras del modelo

- Añadir features de clima (Yr.no API) para Eliteserien
- Añadir rotación europea como feature (Europa League, Conference League)
- Añadir lesiones como feature desde Transfermarkt scraping
- Re-evaluar Eredivisie con ensemble calibrado y más features

### Nuevas ligas (después de Hito 3)

- **Eredivisie**: excluida de Fase 1 por ROI -8.59%, pero promisoria con más features
- **Eliteserien** (Noruega): añadir en junio cuando arranque la temporada (counter-cyclical)
- **EFL League One**: si Betplay la lista consistentemente

### Nuevos mercados (después de 200 apuestas liquidadas)

- BTTS (Both Teams To Score)
- Hándicap asiático ±0.25/±0.75
- Over/Under 3.5

### Infraestructura

- VPS propio (~€5/mes Hetzner) para reducir dependencia de GitHub Actions
- news_monitor.py: scraping de RSS de Kicker y Transfermarkt (el agente ya cubre esto via web search)
- tracker.py: dashboard Streamlit para auditoría semanal

---

## 21. Decisiones de diseño

| Decisión | Elegida | Descartada | Razón |
|---|---|---|---|
| Base de datos | SQLite | PostgreSQL | Cero infraestructura, funciona en Actions |
| Scraping de odds | Kambi API REST | Playwright | Sin auth, más estable, más rápido |
| Scheduled jobs | GitHub Actions | VPS + cron | Costo cero, sin servidor |
| DB entre workflows | Artifacts | Git commit | No contamina historial |
| Ensemble weights | Configurable en YAML | Hardcoded | Permite ajustar sin tocar código |
| Comunicación usuario | Telegram inline buttons | Email / SMS | Respuesta en segundos desde móvil |
| Twitter/X | Web search del agente | API oficial ($100/mes) | Costo cero para Fase 1 |
| Agente IA | Claude Sonnet | GPT-4 | Mejor en análisis de noticias en alemán |

### Por qué no usar Betplay Playwright

Betplay usa Kambi, que expone una API REST pública sin autenticación. Usar Playwright habría sido:
- Más frágil (anti-bot, cambios de UI)
- Más lento (~10x)
- Más costoso (headless browser en cada corrida)

La API de Kambi devuelve el mismo JSON que el frontend, sin restricciones.

### Por qué SQLite y no PostgreSQL

Con un bankroll de 150,000 COP y ~10 apuestas/semana, el volumen de datos es trivial (< 10MB/temporada). PostgreSQL requeriría un servidor adicional y configuración. SQLite corre sin servidor, se puede incluir como artifact en GitHub Actions, y es perfectamente suficiente para este proyecto.

---

## Apéndice A: Literatura académica citada

1. **Dixon, M. J., & Coles, S. G. (1997)**. "Modelling Association Football Scores and Inefficiencies in the Football Betting Market". *Journal of the Royal Statistical Society Series C*.
2. **Angelini, G., & De Angelis, L. (2019)**. "Efficiency of online football betting markets". *International Journal of Forecasting*. 3 de 11 mercados con ineficiencias explotables.
3. **Winkelmann, D., Ötting, M., Deutscher, C., & Makarewicz, T. (2024)**. "Are Betting Markets Inefficient?" *Journal of Sports Economics*. Ineficiencias son de corta duración y no persisten.
4. **Elaad, G. (2020)**. Home advantage efficiency across English tiers.
5. **Kelly, J. L. (1956)**. "A New Interpretation of Information Rate". *Bell System Technical Journal*.

## Apéndice B: Fuentes de datos operativas

| Fuente | Liga | Tipo | Costo | URL |
|---|---|---|---|---|
| football-data.co.uk | BL2, NL1 | CSVs históricos | Gratis | football-data.co.uk |
| Kambi API | BL2 (Betplay) | Odds tiempo real | Gratis | us1.offering-api.kambicdn.com |
| Kicker.de RSS | BL2 | Noticias/lesiones | Gratis | kicker.de/news/rss/feed/2-bundesliga |
| Transfermarkt | BL2 | Lesiones | Gratis (scraping) | transfermarkt.de |
| SofaScore | BL2 | Alineaciones | Gratis (scraping) | sofascore.com |
| Anthropic API | — | Análisis agente | ~$0.35/mes | api.anthropic.com |

## Apéndice C: Checklist pre-apuesta

```
[ ] El agente Claude aprobó la señal
[ ] Overround del mercado < 11%
[ ] No es primera/última 2 jornadas de la liga
[ ] No es post-parón internacional
[ ] No hubo cambio de entrenador reciente
[ ] Movimiento de línea coherente con noticias
[ ] Stake ≤ 5% bankroll (cap duro)
[ ] Exposure total < 15% bankroll (cap duro)
[ ] Durante burn-in: usar 1/8 Kelly (primeras 50 apuestas)
```
