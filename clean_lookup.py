"""
Run ONCE from the folder that contains marathi_vocab_lookup.json:

    python clean_lookup.py

It makes a backup (marathi_vocab_lookup.json.bak), removes single-word
entries whose suggestions are not real synonyms (alignment noise from
difflib), tidies a few entries, and adds missing entries such as "खूप".
Only single-word keys matter to the app, because it looks up one token
at a time.
"""
import json
import shutil

PATH = "marathi_vocab_lookup.json"
shutil.copy(PATH, PATH + ".bak")

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

# Keys whose suggestions are wrong, ungrammatical or not synonyms.
REMOVE = [
    "आले", "करण्यात", "काही", "खरे", "जरी", "जात", "तरी", "तिला", "पडत",
    "पूर्ण", "मन", "मला", "माझे", "वेळ", "वाटले", "वाढते", "सांगता",
    "सांगण्यात", "सुचली", "भाग", "मोठा", "मोठी", "सैल", "जितकं", "तितकं",
    "तितकी", "गेली", "झाले", "होतो", "येते", "शक्य", "राहत", "वाढतात",
    "वाढली", "पाहायला", "खाल्ली", "म्हटले", "ढकलली", "गाव", "सर्वांची",
    "शहरांची", "उडवला",
]

# Keys to rewrite or add (clean 1-2 word synonyms only).
REPLACE = {
    "खूप": ["अत्यंत", "अतिशय", "फार"],
    "दिवस": ["दिन"],
    "वारा": ["पवन", "समीर"],
    "आला": ["दाखल झाला"],
    "गेला": ["हजर झाला"],
    "आधी": ["प्रथम"],
    "एकत्र": ["एकत्रितपणे"],
    "शांत": ["शांतपणे"],
    "चांगला": ["उत्कृष्ट", "आनंददायी"],
    "चांगले": ["उत्कृष्ट"],
    "सोपे": ["सहज", "सुकर"],
    "स्वच्छ": ["नीटनेटके", "निर्मळ", "निरभ्र"],
    "हुशार": ["बुद्धिमान"],
    "नवीन": ["अभिनव", "नूतन"],
    "उंच": ["उत्तुंग"],
    "धाडसी": ["निर्भय", "साहसी"],
    "नम्र": ["विनम्र"],
    "मेहनती": ["कष्टाळू", "जिद्दी"],
    "सुंदर": ["मनमोहक", "आकर्षक", "रमणीय"],
    "छान": ["आल्हाददायक", "रमणीय"],
    "कठीण": ["आव्हानात्मक", "खडतर"],
}

removed = [k for k in REMOVE if data.pop(k, None) is not None]
data.update(REPLACE)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

print(f"Removed {len(removed)} entries, set {len(REPLACE)} entries.")
print(f"Total keys now: {len(data)}  (backup saved as {PATH}.bak)")