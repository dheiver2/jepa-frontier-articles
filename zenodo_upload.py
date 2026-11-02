"""
Batch upload of the 10 articles to Zenodo as DRAFT depositions.

- Reads the token from the ZENODO_TOKEN environment variable (never from a file).
- Creates one draft per article, uploads its PDF and sets the metadata.
- Does NOT publish. Publishing is a separate step (--publish) and is irreversible:
  a published record gets a permanent DOI and cannot be deleted.
- Keeps zenodo_drafts.json so a second run does not create duplicates.

Usage:
  ZENODO_TOKEN=... python3 zenodo_upload.py --check     # read-only token test
  ZENODO_TOKEN=... python3 zenodo_upload.py             # create/update drafts
  ZENODO_TOKEN=... python3 zenodo_upload.py --publish   # publish the drafts (irreversible)
"""
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://zenodo.org/api"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "zenodo_drafts.json")

CREATOR = {
    "name": "Santos, Dheiver Francisco",
    "affiliation": "Independent Researcher",
    "orcid": "0000-0002-8599-9436",
}


def token():
    t = os.environ.get("ZENODO_TOKEN", "").strip()
    if not t:
        sys.exit("ZENODO_TOKEN is not set")
    return t


def call(method, url, data=None, raw=None, ctype="application/json"):
    headers = {"Authorization": f"Bearer {token()}"}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        body = raw
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> HTTP {e.code}: {e.read().decode()[:500]}")


def latex_to_text(s):
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("--", "-").replace("~", " ")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def metadata(tex_path):
    s = open(tex_path, encoding="utf-8").read()
    title = latex_to_text(re.search(r"\\title\{(.*?)\}\n", s, re.S).group(1))
    abstract = latex_to_text(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1))
    kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", s, re.S).group(1)
    keywords = [k.strip() for k in latex_to_text(kw).split(",") if k.strip()]
    note = ("Preprint. Results in this version come from small-scale pilot experiments; "
            "the full experimental protocol is described in the text and is pending.")
    return {
        "upload_type": "publication",
        "publication_type": "preprint",
        "title": title,
        "creators": [CREATOR],
        "description": f"<p>{abstract}</p><p><em>{note}</em></p>",
        "keywords": keywords,
        "access_right": "open",
        "license": "cc-by-4.0",
        "language": "eng",
    }


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def check():
    r = call("GET", f"{API}/deposit/depositions?size=1")
    print(f"token OK; existing depositions visible: {len(r)}")


def create_drafts():
    st = load_state()
    for tex in sorted(glob.glob(os.path.join(HERE, "artigo_*.tex"))):
        name = os.path.splitext(os.path.basename(tex))[0]
        pdf = os.path.join(HERE, name + ".pdf")
        if not os.path.exists(pdf):
            print(f"skip {name}: PDF missing")
            continue
        if name in st:
            dep_id = st[name]["id"]
            dep = call("GET", f"{API}/deposit/depositions/{dep_id}")
            if dep.get("submitted"):
                print(f"skip {name}: already published")
                continue
        else:
            dep = call("POST", f"{API}/deposit/depositions", data={})
            dep_id = dep["id"]
        bucket = dep["links"]["bucket"]
        with open(pdf, "rb") as f:
            call("PUT", f"{bucket}/{name}.pdf", raw=f.read(), ctype="application/octet-stream")
        dep = call("PUT", f"{API}/deposit/depositions/{dep_id}", data={"metadata": metadata(tex)})
        st[name] = {"id": dep_id, "draft_url": dep["links"]["html"],
                    "reserved_doi": dep["metadata"].get("prereserve_doi", {}).get("doi")}
        save_state(st)
        print(f"draft {name}: {st[name]['draft_url']}")
    print(f"\n{len(st)} drafts recorded in {STATE}. Nothing was published.")


def publish():
    st = load_state()
    for name, info in st.items():
        r = call("POST", f"{API}/deposit/depositions/{info['id']}/actions/publish")
        info["doi"] = r.get("doi")
        info["record_url"] = r.get("links", {}).get("record_html")
        save_state(st)
        print(f"published {name}: doi {info['doi']}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--publish" in sys.argv:
        publish()
    else:
        create_drafts()
