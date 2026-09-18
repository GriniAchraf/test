from dav_runoff.validation_figee import dates_regimes, analyser_drift
from dav_runoff.ingestion import load
from dav_runoff.preprocessing import build_panel

panel = build_panel(load(etude, bor3m, **COLONNES), Config(level_relative=True))
reg = dates_regimes(panel)      # → debut_montee, passage_positif, sommet, debut_descente, lus sur ta courbe BOR3M
print(reg)

# Test 1 : figé au début de la montée
d1 = analyser_drift(etude, bor3m, "dav_out_full_rr_v13", date_coupure=reg["debut_montee"], modele="M1", **COLONNES)
# Test 2 : figé au début de la descente
d2 = analyser_drift(etude, bor3m, "dav_out_full_rr_v13", date_coupure=reg["debut_descente"], modele="M1", **COLONNES)
