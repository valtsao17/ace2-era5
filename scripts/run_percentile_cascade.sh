#!/bin/bash
# Regenerate the percentile/temperature-extreme plot chain after the #5 (no-LOYO)
# seasonal_jja_skill recompute. Runs in dependency order.
set -e
cd /home/vt55/ace2
PY=/home/vt55/mm/envs/ace2-fresh/bin/python
L=outputs/lag_may/regen_logs
say(){ echo "===== $(date +%H:%M:%S)  $* ====="; }

say "1/8 seasonal_modkendall_maps";        $PY scripts/seasonal_modkendall_maps.py
say "2/8 precision_recall (global, 22GB)";  $PY scripts/precision_recall_jja_sliding7d.py
say "3/8 cluster sweep --conus";            $PY scripts/cluster_skill_analysis_sliding7d.py --conus
say "4/8 cluster sweep --conus modkendall"; $PY scripts/cluster_skill_analysis_sliding7d.py --conus --metric modkendall
say "5/8 cluster_maps_optimal_k (k=8)";     $PY scripts/cluster_maps_optimal_k.py
say "6/8 precision_recall_clusters";        $PY scripts/precision_recall_clusters.py
say "7/8 skill_panel_combined";             $PY scripts/skill_panel_combined.py
say "8/8 freq_vs_rankcorr_panel";           $PY scripts/freq_vs_rankcorr_panel.py
say "CASCADE DONE"
