"""Serve the interactive Vietnamese POS demo on a loopback address."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlencode, urlsplit

from scripts.demo_language import ContextualTrigram, WordSegmenter, demo_training
from scripts.demo_tv2 import fit_models, load_demo_corpus
from scripts.final_experiment import ROOT, read_lock
from src.data_loader import load_conllu, read_conllu


DEV_EXAMPLES = ("dev-804", "dev-900")
MODE_LABELS = {"best": "Tốt nhất trên dev", "fixed": "Ablation cùng tham số"}
TAG_NAMES = {
    "NOUN": "Danh từ", "VERB": "Động từ", "ADJ": "Tính từ", "ADV": "Phó từ",
    "PRON": "Đại từ", "PROPN": "Tên riêng", "PUNCT": "Dấu câu", "NUM": "Số từ",
    "ADP": "Giới từ", "AUX": "Trợ động từ", "CCONJ": "Liên từ đẳng lập",
    "SCONJ": "Liên từ phụ thuộc", "DET": "Từ hạn định", "PART": "Tiểu từ",
    "INTJ": "Thán từ", "SYM": "Ký hiệu", "X": "Khác",
}
ASSETS = Path(__file__).with_name("demo_assets")


@dataclass
class Example:
    label: str
    value: str
    input_mode: str = "auto"
    gold: list[str] | None = None


def annotated_example(label, tokens, gold):
    return Example(label, " | ".join(tokens), "manual", list(gold))


def verified_split(path: Path, expected_hash: str, name: str) -> Path:
    if sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Tệp {name} lệch mã băm trong hồ sơ đã khóa.")
    return path


class Demo:
    def __init__(self, train_path=None, dev_path=None):
        *train, toy = load_demo_corpus()
        self.examples = {
            "danh-tu": Example("Bàn là đồ vật", "Tôi thấy cái bàn rất đẹp."),
            "dong-tu": Example("Bàn công việc", "Chúng tôi đang bàn về dự án."),
            "toy": annotated_example("Mẫu đồ chơi", *toy),
        }
        params = {"alpha": 0.1, "cutoff": 1}
        self.configs = {"best": (params, params)}
        self.source = "Corpus đồ chơi"
        if train_path:
            candidate = read_lock(ROOT / "results/tables/final_lock.json")["candidate"]
            dataset = candidate["dataset"]
            train = load_conllu(verified_split(train_path, dataset["train_sha256"], "train"))
            if dev_path:
                for s in read_conllu(verified_split(dev_path, dataset["dev_sha256"], "dev")):
                    if s.sent_id in DEV_EXAMPLES:
                        self.examples[s.sent_id] = annotated_example(s.sent_id, s.tokens, s.tags)
            models = candidate["models"]
            self.configs = {
                mode: (models["h1_" + mode]["params"], models["h2_" + mode]["params"])
                for mode in MODE_LABELS
            }
            self.source = "UD Vietnamese VTB"

        self.originals = {
            mode: fit_models(train, bigram_params=bigram, trigram_params=trigram)
            for mode, (bigram, trigram) in self.configs.items()
        }
        supplements = demo_training()
        extended = train + supplements
        self.contextual = ContextualTrigram(alpha=0.1, cutoff=1).fit(extended)
        self.segmenter = WordSegmenter(extended)
        self.n_train = f"{len(train):,}".replace(",", ".")
        self.n_extra = len(supplements)
        self.references = {
            tuple(sample.value.split(" | ")): sample for sample in self.examples.values() if sample.gold
        }
        self.template = Template(Path(__file__).with_name("demo_web.html").read_text(encoding="utf-8"))

    def render(self, raw, manual, input_mode, mode):
        mode = mode if mode in self.configs else "best"
        error = ""
        if input_mode == "auto":
            tokens = self.segmenter.split(raw)
            manual = " | ".join(tokens) if manual is None else manual
        elif input_mode == "manual":
            tokens = [part.strip() for part in manual.split("|")]
            raw = " ".join(tokens) if raw is None else raw
        else:
            tokens = []
            error = "Chọn một trong hai cách nhập rồi gán nhãn lại."
            input_mode = "auto"
        if not tokens or len(tokens) > 100 or any(not token or len(token) > 100 for token in tokens):
            error = error or "Nhập từ 1 đến 100 token, mỗi token tối đa 100 ký tự. Ở chế độ thủ công, điền đủ từ giữa các dấu |."

        reference = self.references.get(tuple(tokens))
        groups = []
        rows = []
        scores = ""
        status = "Đang chờ câu hợp lệ."
        if not error:
            models = self.originals[mode]
            enhanced = self.contextual.predict(tokens)
            original = [model.predict(tokens) for model in models]
            predictions = [enhanced, *original]
            status = f"{len(tokens)} token"
            if reference:
                status += f" · Nhãn chuẩn từ {reference.label}"
                scores = "Đúng theo nhãn chuẩn: " + " · ".join(
                    f"{name} {sum(a == b for a, b in zip(tags, reference.gold, strict=True))}/{len(tokens)}"
                    for name, tags in zip(("Ngữ cảnh", "B0", "Bigram", "Trigram"), predictions, strict=True)
                )
            else:
                status += " · Câu tự nhập chưa có nhãn chuẩn"
            for i, (token, tag) in enumerate(zip(tokens, enhanced, strict=True)):
                word = escape(token)
                tag_name = TAG_NAMES[tag]
                groups.append(
                    f'<li class="annotation t-{tag}" title="{tag_name}"><b>{word}</b>'
                    f'<span>{tag}</span><small>{tag_name}</small></li>'
                )
                gold = reference.gold[i] if reference else None
                cells = []
                for tags in predictions:
                    wrong = gold is not None and tags[i] != gold
                    cls = ' class="incorrect"' if wrong else ""
                    marker = '<span aria-label="khác nhãn chuẩn"> ≠</span>' if wrong else ""
                    cells.append(f"<td{cls}>{tags[i]}{marker}</td>")
                oov = "Có" if token not in models[0].train_vocab_ else "Không"
                rows.append(f'<tr><th scope="row">{word}</th><td>{gold or "—"}</td>'
                            + "".join(cells) + f"<td>{oov}</td></tr>")

        links = []
        for key, example in self.examples.items():
            query = escape(urlencode({"example": key, "mode": mode}))
            links.append(f'<a href="/?{query}">{escape(example.label)}</a>')
        bigram, trigram = self.configs[mode]
        options = "".join(f'<option value="{key}"{" selected" if key == mode else ""}>'
                          f'{MODE_LABELS[key]}</option>' for key in self.configs)
        html = self.template.substitute(
            source=self.source, n_train=self.n_train, n_extra=self.n_extra,
            auto_checked="checked" if input_mode == "auto" else "",
            manual_checked="checked" if input_mode == "manual" else "",
            raw=escape(raw or ""), manual=escape(manual or ""), error=escape(error),
            annotated="".join(groups), status=escape(status), rows="".join(rows),
            scores=escape(scores), examples="".join(links), options=options,
            bigram_config=f"α = {bigram['alpha']}, cutoff = {bigram['cutoff']}",
            trigram_config=f"α = {trigram['alpha']}, cutoff = {trigram['cutoff']}",
        )
        return (400 if error else 200), html.encode("utf-8")


def handler_for(demo):
    class DemoHandler(BaseHTTPRequestHandler):
        def reply(self, code, body, mime="text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if "html" in mime else "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/bricolage.ttf":
                self.reply(200, (ASSETS / "bricolage.ttf").read_bytes(), "font/ttf")
                return
            if request.path != "/":
                self.send_error(404)
                return
            query = parse_qs(request.query)
            sample = demo.examples.get(query.get("example", [""])[0], demo.examples["danh-tu"])
            is_raw = sample.input_mode == "auto"
            self.reply(*demo.render(sample.value if is_raw else None, None if is_raw else sample.value,
                                    sample.input_mode, query.get("mode", ["best"])[0]))

        def do_POST(self):
            if urlsplit(self.path).path != "/":
                self.send_error(404)
                return
            length = self.headers.get("Content-Length", "")
            if not length.isdecimal() or not 0 < int(length) <= 160_000:
                self.send_error(400)
                return
            try:
                form = parse_qs(self.rfile.read(int(length)).decode("utf-8"), keep_blank_values=True)
            except UnicodeDecodeError:
                self.send_error(400)
                return
            input_mode = form.get("input_mode", ["manual"])[0]
            raw = form.get("text", [""])[0] if input_mode == "auto" else None
            manual = form.get("tokens", [""])[0] if input_mode == "manual" else None
            self.reply(*demo.render(raw, manual, input_mode, form.get("mode", ["best"])[0]))

    return DemoHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--train", type=Path, help="Tệp train CoNLL-U đã ghim")
    parser.add_argument("--dev", type=Path, help="Tệp dev để xem ví dụ có nhãn chuẩn")
    args = parser.parse_args()
    if args.dev and not args.train:
        parser.error("--dev cần đi cùng --train.")
    try:
        demo = Demo(args.train, args.dev)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(demo)) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
