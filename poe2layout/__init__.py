"""PoE2 layout capture. Films zone visits, stitches the Tab map of each into a saved run, and groups the runs into
layouts.

Words used throughout the code, the saved files and the README:
    zone        a named place in the game ("Clearfell"); zone_key() is the form we compare
    zone code   the game's id for a zone ("G1_2")
    instance    one generated copy of a zone: zone code + seed (Instance); its id looks like "G1_2_1628456714"
    visit       one stay in an instance, filmed into one video (a town trip only pauses it)
    run         what we save for an instance: map, route and labels (runs/<zone>/<instance id>/)
    label       text shown on the Tab map (Waypoint, Checkpoint, gate names, points of interest)
    gate        a label naming the zone a passage leads to
    piece       part of a map whose position against the rest got lost (merged back once it fits)
    layout      a group of runs with the same map (letters A, B, ...), kept in runs/<zone>/layouts.json
"""
