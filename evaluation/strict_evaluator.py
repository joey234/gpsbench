"""GPSBench v2 diagnostic scorer. Standard library only; v1 is never modified.

See SCORING_SPEC_V2.md for extraction, units, precision, and task definitions.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from collections import Counter

VERSION = "gpsbench-v2.2-all-tasks"
NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
NUM_RE = re.compile(r"(?<![\w.])" + NUMBER + r"(?![\w.])")
EXCLUDED = set()
FORMAT_TOLERANCE = 5e-5


@dataclass(frozen=True)
class Outcome:
    correct: bool
    parsed: bool
    reason: str
    error: float | None = None
    predicted: object = None


def normalize(text):
    # NFKC decomposes double-prime into two single primes; preserve seconds.
    text = unicodedata.normalize("NFKC", str(text).replace("″", '"')).replace("−", "-")
    text = text.replace("\\times", "×").replace("\\cdot", "×")
    text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\,", " ").replace("\\!", "").replace("\\circ", "°")
    text = re.sub(r"\^\{?\s*°\s*\}?", "°", text)
    text = text.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    text = text.replace("**", "").replace("`", "").replace("$", "")
    text = text.replace("′", "'").replace("’", "'").replace("″", '"').replace("“", '"').replace("”", '"')
    return text.strip()


def answer_of(response):
    """Use one explicit final payload; contradictory repeated markers are invalid.

    Without a marker only a direct response (one line, <=300 characters) is used.
    No scanning reasoning for its last number. A multiline final payload is kept.
    """
    text = normalize(response)
    matches = list(re.finditer(r"\bFINAL\s+ANSWER\s*:\s*", text, re.I))
    if matches:
        payloads = [text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
                    for i,m in enumerate(matches)]
        # A Markdown "Final Answer:" heading followed immediately by the
        # required marker is an empty heading, not a second answer.
        payloads = [p for p in payloads if p]
        if not payloads:return None
        if len({re.sub(r"\s+", " ", p).casefold() for p in payloads}) != 1:
            return None
        text = payloads[-1]
    elif "\n" in text or len(text)>300:
        return None
    text = re.sub(r"^\\boxed\{(.*)\}\s*$", r"\1", text, flags=re.S)
    return text.strip() or None


def numbers(text):
    text = normalize(text)
    # Do not parse the squared-unit exponent as a second answer number.
    text = re.sub(r"\b(km|m)\s*(?:\^\s*\{?2\}?|²|2)(?=\W|$)", r"\1²", text)
    scientific = re.compile("("+NUMBER+r")\s*(?:×|x|\*)\s*10\s*\^\s*\{?([-+]?\d+)\}?")
    def expand(m):
        try:
            value=float(m[1].replace(",", ""))*10.**int(m[2])
            return repr(value) if math.isfinite(value) else "INVALID"
        except (OverflowError,ValueError):return "INVALID"
    text=scientific.sub(expand,text)
    def power(m):
        try:
            value=float('1e'+m[1])
            return repr(value) if math.isfinite(value) else 'INVALID'
        except ValueError:return 'INVALID'
    text=re.sub(r'(?<![\w.])10\s*\^\s*\{?([-+]?\d+)\}?',power,text)
    if 'INVALID' in text or re.search(r'(?<!\w)(?:nan|inf(?:inity)?)(?!\w)',text,re.I):return []
    vals=[float(m[0].replace(",", "")) for m in NUM_RE.finditer(text)]
    return vals if all(math.isfinite(v) for v in vals) else []


def scalar(text, kind):
    """Exactly one finite quantity. Units are normalized to km/km²/degrees."""
    if text is None:return None
    text=normalize(text)
    # Area/perimeter answers may explicitly label two different requested quantities.
    if kind=="area" and re.search(r"\barea\s*[:=]",text,re.I):
        if len(re.findall(r"\barea\s*[:=]",text,re.I)) != 1:return None
        text=re.split(r"\barea\s*[:=]",text,flags=re.I)[1]
        text=re.split(r"\bperimeter\b",text,flags=re.I)[0]
    vals=numbers(text)
    if len(vals)!=1:return None
    value=vals[0];unit=text.casefold()
    if kind=="distance":
        if re.search(r"\b(?:miles?|mi)\b",unit):value*=1.609344
        elif re.search(r"\b(?:meters?|metres?|m)\b",unit) and not re.search(r"\bkm\b",unit):value/=1000
        elif re.search(r"\b(?:feet|ft|yards?|yd|nautical)\b",unit):return None
    elif kind=="area":
        if re.search(r"\b(?:m\s*(?:²|\^\{?2\}?|2)|square meters?|square metres?)",unit):value/=1e6
        elif re.search(r"\b(?:miles?|mi|feet|ft|hectares?)\b",unit):return None
    elif kind=="bearing":
        if re.search(r"\brad(?:ians?)?\b",unit):value=math.degrees(value)
    return value if math.isfinite(value) else None


def coordinates(text, geographic=True, target_format=None):
    if text is None:return None
    text=normalize(text)
    text=re.sub(r"(?<=\d)''",'"',text)
    dms=re.compile(r"("+NUMBER+r")\s*°\s*(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)\s*\"\s*([NSEW])",re.I)
    parts=list(dms.finditer(text))
    if parts:
        if len(parts)!=2 or target_format=="decimal_degrees":return None
        axes={}
        for m in parts:
            deg=float(m[1]);mins=float(m[2]);secs=float(m[3]);hem=m[4].upper()
            # The release contains 26 strings rounded to exactly 60.0 seconds
            # without carrying. Normalize that boundary arithmetically.
            if mins>=60 or secs>60 or (deg<0 and hem in "NE"):return None
            value=(abs(deg)+mins/60+secs/3600)*(-1 if hem in "SW" else 1)
            key="lat" if hem in "NS" else "lon"
            if key in axes:return None
            axes[key]=value
        if set(axes)!={"lat","lon"}:return None
        vals=[axes["lat"],axes["lon"]]
        # Any additional numerical answer outside the two DMS expressions is ambiguous.
        remainder=dms.sub("",text)
        if numbers(remainder):return None
    else:
        if target_format=="DMS":return None
        if "°" in text and ("'" in text or '"' in text):return None
        # Projection zone/CRS metadata is not an easting or northing.
        if not geographic:
            text=re.sub(r"\b(?:UTM\s*)?zone\s*\d{1,2}\s*[NS]?", "", text, flags=re.I)
            text=re.sub(r"\bEPSG\s*:?\s*\d+", "", text, flags=re.I)
        vals=numbers(text)
        if len(vals)!=2:return None
        if geographic:
            decimal_hemi=re.findall(r"("+NUMBER+r")\s*°?\s*([NSEW])\b",text,re.I)
            if decimal_hemi:
                if len(decimal_hemi)!=2:return None
                axes={}
                for number,hem in decimal_hemi:
                    hem=hem.upper();value=float(number.replace(",", ""))
                    if value<0 and hem in "NE":return None
                    key="lat" if hem in "NS" else "lon"
                    if key in axes:return None
                    axes[key]=abs(value)*(-1 if hem in "SW" else 1)
                if set(axes)!={"lat","lon"}:return None
                vals=[axes["lat"],axes["lon"]]
    if geographic and not (-90<=vals[0]<=90 and -180<=vals[1]<=180):return None
    return tuple(vals)


def name_key(text):
    text=normalize(text).casefold().strip(" .\"'[]()")
    return re.sub(r"\s+", " ", text)


def choose_name(text,names):
    """One complete candidate name, with optional units or explanation suffix.

    Multiple known names anywhere in a final payload are rejected. Word boundaries
    prevent substrings such as York matching Yorkshire. Accents remain significant.
    """
    if text is None:return None
    key=name_key(text)
    found=[n for n in names if re.search(r"(?<!\w)"+re.escape(name_key(n))+r"(?!\w)",key)]
    if len(set(map(name_key,found)))!=1:return None
    name=found[0];nk=name_key(name)
    if not (key==nk or key.startswith(nk+",") or key.startswith(nk+" (")
            or key.startswith(nk+" is ") or key.startswith(nk+" -") or key.startswith(nk+".")):
        return None
    # Alternatives not known in the candidate set must not silently pass.
    suffix=key[len(nk):]
    if re.search(r"\b(?:or|alternatively|maybe|possibly)\b|/",suffix):return None
    return name


def boolean(text):
    if text is None:return None
    key=name_key(text)
    tokens=re.findall(r"\b(?:yes|no|true|false|correct|incorrect)\b",key)
    positives={"yes","true","correct"}; negatives={"no","false","incorrect"}
    if not tokens or (any(t in positives for t in tokens) and any(t in negatives for t in tokens)):
        return None
    if not re.match(r"^(?:yes|no|true|false|correct|incorrect)\b",key):return None
    return tokens[0] in positives


def letter(text,allowed):
    if text is None:return None
    match=re.fullmatch(r"\s*\(?([A-Z])\)?(?:[.)]|\s*[-:]\s*[^\n]+)?\s*",normalize(text),re.I)
    if not match:return None
    val=match[1].upper()
    if val not in allowed:return None
    # A letter plus an alternative letter is not one selection.
    if re.search(r"\b(?:or|and)\s+[A-Z]\b",text,re.I):return None
    return val


def axis_direction(source):
    """Repository coordinate-axis convention, from point B to point A.

    Compare absolute differences in degrees; wrap longitude into [-180,180].
    Latitude wins ties. This is deliberately distinct from an initial azimuth.
    """
    a,b=source['point_a'],source['point_b']
    lat=a['lat']-b['lat'];lon=a['lon']-b['lon']
    if lon>180:lon-=360
    elif lon < -180:lon+=360
    if abs(lat)>=abs(lon):return 'North' if lat>0 else 'South'
    return 'East' if lon>0 else 'West'


def candidate_targets(source,gt):
    """Indices identifiable from the fields displayed in the actual prompt.

    Identical name/country/population entries are observationally equivalent.
    Accept any such entry, rather than inventing an unobservable identity.
    """
    return [i for i,c in enumerate(source['candidates'])
            if all(c.get(k)==gt.get(k) for k in ('name','country','population'))]


CONTINENT_PATTERN = re.compile(
    r'(?<!\w)(north america|south america|antarctica|africa|asia|europe|oceania|australia)(?!\w)', re.I)


def continent_key(text):
    value=name_key(text)
    return 'oceania' if value=='australia' else value


def grouping_of(answer, names):
    """Extract explicit continent groups or city-to-continent assignments.

    Accept multiline/semicolon groups, bullet lists, and city-first rows.
    Require every requested city once, with one assigned continent. Mere
    continent mentions, incomplete lists, and contradictory assignments fail.
    """
    text=normalize(answer).casefold()
    name_lookup={name_key(n):n for n in names}
    city_pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(n) for n in sorted(name_lookup,key=len,reverse=True))+r')(?!\w)')
    if Counter(name_lookup[m[0]] for m in city_pattern.finditer(text))!=Counter(names):return None
    # A new explicit header can start within a semicolon group or sentence.
    header=re.compile(r'(?<!\w)(?:north america|south america|antarctica|africa|asia|europe|oceania|australia)(?:\s*/\s*(?:australia|oceania))?(?:\s*\((?:australia|oceania)\))?\s*[:—–-]',re.I)
    text=header.sub(lambda m:'\n'+m[0],text)
    assigned=[];current=None
    for segment in re.split(r'[\n;|]+',text):
        cities=[(m.start(),m.end(),name_lookup[m[0]]) for m in city_pattern.finditer(segment)]
        continents=[(m.start(),m.end(),continent_key(m[0])) for m in CONTINENT_PATTERN.finditer(segment)]
        # A continent word inside a city name is not a group heading.
        continents=[c for c in continents if not any(a<=c[0]<b for a,b,_ in cities)]
        parentheses=[m.span() for m in re.finditer(r'\([^()]*\)',segment)]
        # Country qualifiers such as '(South Africa)' are not a group label.
        continents=[c for c in continents if not any(a<=c[0]<b for a,b in parentheses)]
        if not cities:
            if continents:
                unique={c[2] for c in continents}
                if len(unique)==1:current=next(iter(unique))
                else:current=None
            continue
        if continents and continents[0][0] < cities[0][0]:
            # Continent-first groups; a following header changes the group.
            for start,end,n in cities:
                before=[c for c in continents if c[0]<start]
                if not before:return None
                assigned.append((n,before[-1][2]))
            current=continents[-1][2]
        elif continents:
            # A city list may share a trailing continent: 'A, B - Europe'.
            distinct={c[2] for c in continents}
            if len(distinct)==1 and continents[0][0]>=cities[-1][1]:
                assigned.extend((n,next(iter(distinct))) for _,_,n in cities)
            else:
                for i,(start,end,n) in enumerate(cities):
                    next_city=cities[i+1][0] if i+1<len(cities) else len(segment)
                    after={c[2] for c in continents if end<=c[0]<next_city}
                    if len(after)!=1:return None
                    assigned.append((n,next(iter(after))))
            current=None
        elif current is not None:
            assigned.extend((n,current) for _,_,n in cities)
        else:return None
    return assigned if Counter(n for n,_ in assigned)==Counter(names) else None


def score_row(response,gt,task,source):
    answer=answer_of(response)
    if answer is None:return Outcome(False,False,"missing_or_ambiguous_final")
    if task=='relative_position':
        chosen=name_key(answer)
        # Prompts request a direction word, not a letter or a diagonal.
        valid=chosen in {'north','south','east','west'}
        return Outcome(valid and chosen==axis_direction(source).casefold(),valid,'axis_direction',predicted=chosen if valid else None)
    if task=='name_disambiguation':
        candidates=source['candidates'];targets=candidate_targets(source,gt)
        if not targets:raise ValueError('No candidate matches the displayed target metadata')
        chosen=letter(answer,set(chr(65+i) for i in range(len(candidates))))
        if chosen is not None:
            index=ord(chosen)-65
            return Outcome(index in targets,True,'candidate_identity',predicted=index)
        # Direct metadata answers must uniquely resolve a displayed candidate,
        # modulo identical displayed entries. A shared city name is insufficient.
        key=name_key(answer)
        matched=[i for i,c in enumerate(candidates)
                 if name_key(c['name']) in key and re.search(r'(?<!\w)'+re.escape(c['country'].casefold())+r'(?!\w)',key)]
        if len(matched)>1:
            pops=numbers(answer)
            matched=[i for i in matched if candidates[i]['population'] in pops]
        identities={tuple(candidates[i].get(k) for k in ('name','country','population')) for i in matched}
        valid=len(identities)==1 and not re.search(r'\b(?:or|alternatively)\b',key)
        return Outcome(valid and set(matched)<=set(targets),valid,'candidate_metadata',predicted=matched if valid else None)
    if task=='boundary_analysis':
        gold=[(n,continent_key(c)) for c,ns in gt['grouping'].items() for n in ns]
        predicted=grouping_of(answer,[n for n,_ in gold])
        return Outcome(predicted is not None and Counter(predicted)==Counter(gold),predicted is not None,'city_continent_assignments',predicted=predicted)
    if task=='spatial_patterns':
        chosen=choose_name(answer,[p['name'] for p in source['points']])
        return Outcome(chosen is not None and name_key(chosen)==name_key(gt['answer']),chosen is not None,'outlier_identity',predicted=chosen)
    if task=="format_conversion":
        target=source.get("target_format")
        target="decimal_degrees" if str(target).casefold() in {"dd","decimal_degrees"} else "DMS"
        pair=coordinates(answer,target_format=target)
        good=pair is not None and all(abs(p-e)<=FORMAT_TOLERANCE+1e-12 for p,e in zip(pair,[gt["latitude"],gt["longitude"]]))
        return Outcome(good,pair is not None,"coordinate_pair",predicted=pair)
    if task=="coordinate_system_transformation":
        pair=coordinates(answer,geographic=False)
        keys,tol=(["x","y"],100.) if "x" in gt else (["easting","northing"],10.)
        good=pair is not None and all(abs(p-gt[k])<=tol for p,k in zip(pair,keys))
        return Outcome(good,pair is not None,"projected_pair",predicted=pair)
    if task in {"distance_calculation","bearing_computation","area_perimeter"}:
        kind,field=("distance","distance_km") if task=="distance_calculation" else ("bearing","initial_bearing_deg") if task=="bearing_computation" else ("area","area_km2")
        value=scalar(answer,kind);gold=gt[field]
        if value is None or (kind!="bearing" and value<0):return Outcome(False,False,"invalid_quantity")
        absolute=abs(value-gold)
        if kind=="bearing":
            absolute=absolute%360;absolute=min(absolute,360-absolute);error=absolute/180
            tol=gt.get("tolerance_deg",5.)
        else:
            error=absolute/abs(gold) if gold else None
            tol=gt.get("tolerance_km",abs(gold)*.05) if kind=="distance" else abs(gold)*gt.get("tolerance_percent",5)/100
        return Outcome(absolute<=tol,True,"quantity",error,value)
    if task=="interpolation":
        pair=coordinates(answer);gold=gt["interpolated_point"]
        tol=gt.get("tolerance_deg",.01)
        good=pair is not None and abs(pair[0]-gold["lat"])<=tol and abs(pair[1]-gold["lon"])<=tol
        return Outcome(good,pair is not None,"interpolated_pair",predicted=pair)
    if task=="bounding_box":
        vals=numbers(answer);keys=["min_lat","max_lat","min_lon","max_lon"]
        valid=len(vals)==4 and all(-90<=v<=90 for v in vals[:2]) and all(-180<=v<=180 for v in vals[2:])
        return Outcome(valid and all(abs(p-gt[k])<=.5 for p,k in zip(vals,keys)),valid,"bounding_box",predicted=vals)
    if task=="route_geometry":
        chosen=letter(answer,set("ABCD"))
        if chosen is not None:return Outcome(chosen==gt["answer"],True,"mcq",predicted=chosen)
        if re.fullmatch(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",answer):
            vals=[int(x) for x in re.findall(r"\d+",answer)]
            return Outcome(len(set(vals))==len(vals) and set(vals)==set(gt["points_kept"]),True,"indices",predicted=vals)
        return Outcome(False,False,"invalid_route_selection")
    if task=="terrain_classification":
        chosen=letter(answer,set("ABCDEF"))
        return Outcome(chosen==gt["answer"],chosen is not None,"mcq",predicted=chosen)
    if task=="route_analysis":
        pred=boolean(answer)
        return Outcome(pred is not None and pred==gt["answer"],pred is not None,"boolean",predicted=pred)
    if task=="proximity":
        chosen=choose_name(answer,[x["name"] for x in source["candidate_points"]])
        return Outcome(chosen is not None and name_key(chosen)==name_key(gt["answer"]),chosen is not None,"candidate",predicted=chosen)
    if task=="place_association":
        # Score the canonical city field only. A comma-separated city list must
        # not pass: remaining fields must be stored geographic qualifiers.
        key=name_key(answer);expected=name_key(gt["location_name"])
        qualifiers={name_key(gt[k]) for k in ("admin1_name","country_name","country_code") if gt.get(k)}
        if gt.get("country_code")=="US":qualifiers.update({"usa","u.s.","u.s.a.","united states of america"})
        suffix=key[len(expected)+1:] if key.startswith(expected+",") else None
        good=key==expected or (suffix is not None and all(name_key(p) in qualifiers for p in suffix.split(",")))
        parenthesized=re.fullmatch(r'(.*?)\s*\(([^()]*)\)\s*\.?',normalize(answer))
        if parenthesized and not good:
            good=name_key(parenthesized[1])==expected and all(name_key(p) in qualifiers for p in parenthesized[2].split(','))
        return Outcome(good,True,"canonical_city",predicted=key)
    if task in {"missing_data","missing_data_handling"}:
        vals=numbers(answer);field="latitude" if "latitude" in gt else "longitude"
        valid=len(vals)==1 and abs(vals[0])<=(90 if field=="latitude" else 180)
        return Outcome(valid and abs(vals[0]-gt[field])<=gt.get("tolerance",.1),valid,"missing_axis",predicted=vals)
    raise ValueError("Unsupported task: "+task)
