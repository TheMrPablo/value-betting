import sqlite3

c = sqlite3.connect('data/betting.db')
cur = c.cursor()

fixtures = [
    # ROUND 30
    ('BL2', '2025-26', 'Elversberg',        'Karlsruhe',          '2026-04-17 16:30:00', 30),
    ('BL2', '2025-26', 'Holstein Kiel',     'Kaiserslautern',     '2026-04-17 16:30:00', 30),
    ('BL2', '2025-26', 'Bielefeld',         'Nurnberg',           '2026-04-18 11:00:00', 30),
    ('BL2', '2025-26', 'Magdeburg',         'Fortuna Dusseldorf', '2026-04-18 11:00:00', 30),
    ('BL2', '2025-26', 'Dresden',           'Bochum',             '2026-04-18 11:00:00', 30),
    ('BL2', '2025-26', 'Hannover',          'Paderborn',          '2026-04-18 18:30:00', 30),
    ('BL2', '2025-26', 'Braunschweig',      'Hertha',             '2026-04-19 11:30:00', 30),
    ('BL2', '2025-26', 'Greuther Furth',    'Darmstadt',          '2026-04-19 11:30:00', 30),
    ('BL2', '2025-26', 'Schalke 04',        'PreuÃen MÃ¼nster',  '2026-04-19 11:30:00', 30),
    # ROUND 31
    ('BL2', '2025-26', 'Fortuna Dusseldorf','Dresden',            '2026-04-24 16:30:00', 31),
    ('BL2', '2025-26', 'Kaiserslautern',    'Braunschweig',       '2026-04-24 16:30:00', 31),
    ('BL2', '2025-26', 'Hertha',            'Holstein Kiel',      '2026-04-25 11:00:00', 31),
    ('BL2', '2025-26', 'Karlsruhe',         'Hannover',           '2026-04-25 11:00:00', 31),
    ('BL2', '2025-26', 'PreuÃen MÃ¼nster', 'Bielefeld',          '2026-04-25 11:00:00', 31),
    ('BL2', '2025-26', 'Darmstadt',         'Elversberg',         '2026-04-25 18:30:00', 31),
    ('BL2', '2025-26', 'Bochum',            'Greuther Furth',     '2026-04-26 11:30:00', 31),
    ('BL2', '2025-26', 'Nurnberg',          'Magdeburg',          '2026-04-26 11:30:00', 31),
    ('BL2', '2025-26', 'Paderborn',         'Schalke 04',         '2026-04-26 11:30:00', 31),
    # ROUND 32
    ('BL2', '2025-26', 'Bielefeld',         'Bochum',             '2026-05-02 11:00:00', 32),
    ('BL2', '2025-26', 'Holstein Kiel',     'Braunschweig',       '2026-05-02 11:00:00', 32),
    ('BL2', '2025-26', 'Dresden',           'Kaiserslautern',     '2026-05-02 11:00:00', 32),
    ('BL2', '2025-26', 'Schalke 04',        'Fortuna Dusseldorf', '2026-05-02 18:30:00', 32),
    ('BL2', '2025-26', 'Elversberg',        'Paderborn',          '2026-05-03 11:30:00', 32),
    ('BL2', '2025-26', 'Greuther Furth',    'Nurnberg',           '2026-05-03 11:30:00', 32),
    ('BL2', '2025-26', 'Hannover',          'PreuÃen MÃ¼nster',  '2026-05-03 11:30:00', 32),
    ('BL2', '2025-26', 'Karlsruhe',         'Darmstadt',          '2026-05-03 11:30:00', 32),
    ('BL2', '2025-26', 'Magdeburg',         'Hertha',             '2026-05-03 11:30:00', 32),
    # ROUND 33
    ('BL2', '2025-26', 'Kaiserslautern',    'Bielefeld',          '2026-05-08 16:30:00', 33),
    ('BL2', '2025-26', 'Paderborn',         'Karlsruhe',          '2026-05-08 16:30:00', 33),
    ('BL2', '2025-26', 'Bochum',            'Hannover',           '2026-05-09 11:00:00', 33),
    ('BL2', '2025-26', 'Braunschweig',      'Dresden',            '2026-05-09 11:00:00', 33),
    ('BL2', '2025-26', 'Holstein Kiel',     'Magdeburg',          '2026-05-09 11:00:00', 33),
    ('BL2', '2025-26', 'Nurnberg',          'Schalke 04',         '2026-05-09 18:30:00', 33),
    ('BL2', '2025-26', 'Fortuna Dusseldorf','Elversberg',         '2026-05-10 11:30:00', 33),
    ('BL2', '2025-26', 'Hertha',            'Greuther Furth',     '2026-05-10 11:30:00', 33),
    ('BL2', '2025-26', 'PreuÃen MÃ¼nster', 'Darmstadt',          '2026-05-10 11:30:00', 33),
    # ROUND 34
    ('BL2', '2025-26', 'Bielefeld',         'Hertha',             '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Darmstadt',         'Paderborn',          '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Elversberg',        'PreuÃen MÃ¼nster',  '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Greuther Furth',    'Fortuna Dusseldorf', '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Hannover',          'Nurnberg',           '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Karlsruhe',         'Bochum',             '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Magdeburg',         'Kaiserslautern',     '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Schalke 04',        'Braunschweig',       '2026-05-17 13:30:00', 34),
    ('BL2', '2025-26', 'Dresden',           'Holstein Kiel',      '2026-05-17 13:30:00', 34),
]

inserted = 0
for f in fixtures:
    cur.execute("""
        INSERT OR IGNORE INTO fixtures
            (league, season, home_team_name, away_team_name, kickoff_utc, status, matchweek)
        VALUES (?, ?, ?, ?, ?, 'scheduled', ?)
    """, f)
    if cur.rowcount > 0:
        inserted += 1

c.commit()
total = cur.execute("SELECT COUNT(*) FROM fixtures WHERE status='scheduled'").fetchone()[0]
print(f"Insertados: {inserted} | Total scheduled: {total}")

print("\nPróximos 5 fixtures:")
for r in cur.execute("""
    SELECT home_team_name, away_team_name, kickoff_utc, matchweek
    FROM fixtures WHERE status='scheduled'
    ORDER BY kickoff_utc ASC LIMIT 5
""").fetchall():
    print(f"  J{r[3]}: {r[0]} vs {r[1]} — {r[2]}")

c.close()