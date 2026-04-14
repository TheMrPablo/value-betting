import sqlite3, json
c = sqlite3.connect('data/betting.db')
rows = c.execute("SELECT id, flags FROM signals WHERE id IN (1,2)").fetchall()
for r in rows:
    print(f"\n=== Signal {r[0]} ===")
    try:
        data = json.loads(r[1])
        print(f"Análisis: {data.get('agent_analysis', 'N/A')}")
        print(f"Noticias: {data.get('agent_news', 'N/A')}")
        print(f"Confianza: {data.get('agent_confidence', 'N/A')}")
    except:
        print(r[1])