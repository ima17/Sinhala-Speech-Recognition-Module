"""
Defines dictionaries for converting
between text and integer sequences.
"""

char_map_str = """
' 0
<SPACE> 1
අ 2
ආ 3
ඇ 4
ඈ 5
ඉ 6
ඊ 7
උ 8
ඌ 9
ා 10
ැ 11
ෑ 12
ි 13
ී 14
ු 15
ූ 16
එ 17
ඒ 18
ඓ 19
ඔ 20
ඕ 21
ඖ 22
ෙ 23
ේ 24
ෛ 25
ො 26
ෝ 27
ෞ 28
ක 29
ග 30
ච 31
ට 32
ඩ 33
ත 34
ද 35
න 36
ප 37
බ 38
ම 39
ය 40
ර 41
ල 42
ව 43
හ 44
ස 45
් 46

"""

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index) + 1] = ch
index_map[2] = ' '
