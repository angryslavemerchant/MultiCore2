"""Nonce entity name generators for the stream probe.

Names are freshly minted per lifetime so model weights can never supply
the payload; only in-context retrieval can. All generators aim for
multi-token names under the NeoX BPE (2+ tokens) that read as plausible
proper nouns without colliding with real-world entities the model may
have memorized from FineWeb.
"""
import random

_ON = ["vor", "tav", "kre", "zan", "mel", "dro", "fen", "gal", "hux",
       "jor", "lys", "mar", "nev", "os", "pel", "quin", "rho", "syl",
       "tor", "ul", "vex", "wren", "yar", "bel", "cor", "dun"]
_MID = ["a", "e", "i", "o", "u", "ar", "el", "in", "or", "us"]
_END = ["lath", "mar", "rix", "dan", "vek", "los", "net", "gard",
        "wick", "born", "stad", "mund", "ley", "ron", "vis", "tane"]

_SURN = ["Kremar", "Voss", "Hallan", "Drexel", "Marovic", "Tannen",
         "Quillon", "Ferber", "Ostrand", "Belmont", "Ashwick", "Corven",
         "Malbray", "Renholt", "Sarker", "Widmore", "Galt", "Pruett",
         "Novak", "Thorne"]

_CO_SUFFIX = ["Holdings", "Systems", "Labs", "Group", "Industries",
              "Dynamics", "Analytics", "Logistics", "Partners",
              "Technologies", "Works", "Collective"]

_CITY_PRE = ["Port ", "East ", "North ", "Lake ", ""]
_CITY_END = ["dale", "mont", "haven", "brook", "field", "moor", "wick",
             "ford", "gate", "crest", "hollow", "march"]

_PROD_NUM = ["2", "3", "5", "7", "9", "12", "X", "Max", "Mini", "Pro"]


def _stem(rng, syllables=2):
    s = rng.choice(_ON)
    for _ in range(syllables - 1):
        s += rng.choice(_MID) + rng.choice(_END)
    return s.capitalize()


def person(rng: random.Random) -> str:
    return f"{_stem(rng)} {rng.choice(_SURN)}"


def company(rng: random.Random) -> str:
    return f"{_stem(rng)} {rng.choice(_CO_SUFFIX)}"


def city(rng: random.Random) -> str:
    return f"{rng.choice(_CITY_PRE)}{_stem(rng, 1)}{rng.choice(_CITY_END)}"


def product(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return f"the {_stem(rng)} {rng.choice(_PROD_NUM)}"
    return f"the {_stem(rng)}{rng.choice(_END)}"


PAYLOAD_KIND = {"works_at": company, "lives_in": city,
                "makes": product, "based_in": city}
SUBJECT_KIND = {"works_at": person, "lives_in": person,
                "makes": company, "based_in": company}
