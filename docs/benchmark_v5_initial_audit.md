RAW SOURCES FOUND:
| Annotator | Images | JSON | Matched pairs | Orphan JSON | Invalid JSON |
|---|---:|---:|---:|---:|---:|
| Jakub | 62 | 63 | 62 | 1 | 0 |
| Marek | 26 | 26 | 26 | 0 | 0 |
| Matej | 87 | 297 | 87 | 210 | 99 |
| oliver | 166 | 166 | 166 | 0 | 0 |

JAKUB:
images: 62
json: 63
matched pairs: 62
labels: Tree, TreeGroup
shape types: polygon
problems: {'repeated polygon vertex': 4}; orphan JSON: 1

TOTAL MATCHED PAIRS: 341
ELIGIBLE BEFORE DEDUPLICATION: 298
DUPLICATE GROUPS: 49
ANNOTATION CONFLICTS: 39
EXCLUDED: 43 images; {'repeated polygon vertex': 15, 'unsupported/excluded shape type: circle': 1, 'self-intersecting polygon': 26, 'unsupported/excluded label: zmladenie': 1}
EXPECTED UNIQUE IMAGES: PENDING conflict review; 217 uncontested, up to 255 eligible unique candidates
RECOMMENDED SPLIT: 70% / 15% / 15%, seed 42; image-level pending confirmation of parent-image grouping
train: 178 (conditional on retaining all eligible unique candidates)
val: 38 (conditional on retaining all eligible unique candidates)
test: 39 (conditional on retaining all eligible unique candidates)
SAFE TO BUILD BENCHMARK_V5: NO
SPATIAL LEAKAGE STATUS: UNVERIFIED

Filenames suggest tiles from larger images. Confirm parent image/flight identities before a grouped split; names alone do not establish spatial independence.

Per-folder counts below are direct children, so nested sources are not double-counted.
| Folder | Images | JSON | Pairs | Orphan images | Orphan JSON | Invalid JSON | Duplicate images | Labels | Shape types |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| . | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Jakub | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Jakub/1.2.2026_hotovo (1) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Jakub/1.2.2026_hotovo (1)/1.2.2026_hotovo | 31 | 31 | 31 | 0 | 0 | 0 | 24 | Tree, TreeGroup | polygon |
| Jakub/19_4hotove | 31 | 32 | 31 | 0 | 1 | 0 | 0 | TreeGroup, Tree | polygon |
| Marek | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Marek/1.2.2026_tiles_1024 | 26 | 26 | 26 | 0 | 0 | 0 | 24 | Tree, TreeGroup | polygon, circle |
| Matej | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Matej/19_4_2026_plocha1_part2_opravene | 87 | 99 | 87 | 0 | 12 | 0 | 0 | Tree, TreeGroup | polygon |
| Matej/anotations_vojtanik | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Matej/anotations_vojtanik/__MACOSX | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| Matej/anotations_vojtanik/__MACOSX/anotations_vojtanik | 0 | 99 | 0 | 0 | 99 | 99 | 0 |  |  |
| Matej/anotations_vojtanik/anotations_vojtanik | 0 | 99 | 0 | 0 | 99 | 0 | 0 | zmladenie | polygon |
| oliver | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |
| oliver/19_4_2026_plocha1_titles_1024_part1_Oliver_Surovy_NTB | 86 | 86 | 86 | 0 | 0 | 0 | 0 | TreeGroup, Tree | polygon |
| oliver/19_4_2026_plocha1_titles_1024_part1_Oliver_Surovy_PC | 20 | 20 | 20 | 0 | 0 | 0 | 0 | TreeGroup, Tree | polygon |
| oliver/19_4_Plocha_1_part1 | 25 | 25 | 25 | 0 | 0 | 0 | 25 | Tree, TreeGroup, tree | polygon |
| oliver/hotove-stromceky | 35 | 35 | 35 | 0 | 0 | 0 | 25 | Tree, TreeGroup, zmladenie | polygon |
