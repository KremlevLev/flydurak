import argparse, html, math, random
from urllib.parse import parse_qs
import torch, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from .connectome import load_graph
from .durak import DurakGame, PASS, TAKE
from .durak_play import RANK_NAMES, SUIT_NAMES, observation
from .model import DeepFlyPolicy, FlyPolicy

SUIT_WORDS = ("треф", "бубен", "червей", "пик")
_RENDER_COORDINATES, _RENDER_ANATOMY = {}, []

def sort_hand(cards, trump):
    suits = [s for s in range(4) if s != trump] + [trump]
    order = {s: i for i, s in enumerate(suits)}
    return sorted(cards, key=lambda c: (order[c // 9], c % 9))

def card_html(card, clickable=False):
    rank, suit = RANK_NAMES[card % 9], SUIT_NAMES[card // 9]
    classes = "card red" if card // 9 in (1, 2) else "card black"
    body = f'<span class="rank">{rank}</span><span class="suit">{suit}</span><span class="name">{SUIT_WORDS[card // 9]}</span>'
    return f'<button class="{classes} legal" name="action" value="{card}">{body}</button>' if clickable else f'<div class="{classes}">{body}</div>'

def hand_html(cards, legal, trump):
    groups = []
    for current_suit in [s for s in range(4) if s != trump] + [trump]:
        suited = [c for c in sort_hand(cards, trump) if c // 9 == current_suit]
        if not suited:
            continue
        label = f"КОЗЫРИ {SUIT_NAMES[current_suit]}" if current_suit == trump else SUIT_NAMES[current_suit]
        group_class = "suit-group trump-group" if current_suit == trump else "suit-group"
        rendered = "".join(card_html(c, True) if c in legal else card_html(c).replace('class="card ', 'class="card unavailable ') for c in suited)
        groups.append(f'<section class="{group_class}"><div class="suit-label">{label}</div><div class="cards">{rendered}</div></section>')
    return "".join(groups)

def _project_neuron(neuron_id):
    value = (int(neuron_id) * 2654435761 + 2246822519) & 0xFFFFFFFF
    u, v = ((value & 0xFFFF) + .5) / 65536, (((value >> 16) & 0xFFFF) + .5) / 65536
    radius, angle = math.sqrt(u), 2 * math.pi * v
    if value % 100 < 42:
        return 168 + 58 * radius * math.cos(angle), 125 + 96 * radius * math.sin(angle)
    return 414 + 132 * radius * math.cos(angle), 125 + 51 * radius * math.sin(angle)

def activity_svg(activity, edges=None, history=None, coordinates=None, anatomy=None):
    coordinates = coordinates if coordinates is not None else _RENDER_COORDINATES
    anatomy = anatomy if anatomy is not None else _RENDER_ANATOMY
    if not activity:
        return '<div class="activity-empty">Активность появится после первого хода мухи.</div>'
    if not coordinates or not anatomy:
        return '<div class="activity-empty">В графе нет настоящих somaLocation. Выполни <code>flysans-prepare positions</code>; искусственная геометрия отключена.</div>'
    peak = max(abs(value) for _, value in activity) or 1.0
    background = [f'<circle cx="{x:.1f}" cy="{y:.1f}" r=".95" class="background-neuron"/>' for x, y in anatomy]
    links = []
    edges = edges or []
    edge_peak = max((abs(edge[2]) for edge in edges), default=1.0) or 1.0
    for source, target, signal, weight in edges:
        if source not in coordinates or target not in coordinates: continue
        x1, y1 = coordinates[source]; x2, y2 = coordinates[target]
        strength = min(1.0, abs(signal) / edge_peak)
        color = "#ffb84d" if signal >= 0 else "#27cfff"
        links.append(f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{color}" stroke-width="{.25+2.3*strength:.2f}" stroke-opacity="{.06+.55*strength:.2f}" marker-end="url(#arrow)" class="activity-link"><title>{source} → {target}; signal {signal:+.4f}; weight {weight:+.4f}</title></path>')
    dots = []
    for order, (neuron_id, value) in enumerate(reversed(activity)):
        if neuron_id not in coordinates: continue
        x, y = coordinates[neuron_id]
        strength = min(1.0, abs(value) / peak)
        color = "#ffe5a3" if value >= 0 else "#b8f1ff"
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.15" fill="{color}" fill-opacity="{.82 + .18 * strength:.2f}" class="active-neuron" style="--arrival:{.18 + (order % 24) * .055:.2f}s"><title>neuron {neuron_id}: {value:+.4f}</title></circle>')
    history = history or []
    history_bars = "".join(f'<i style="height:{max(8, min(100, sum(abs(v) for _,v in frame)/(len(frame) or 1)*115)):.0f}%"></i>' for frame in history)
    return f'''<div class="cns-stage"><svg class="brain-map" viewBox="0 0 610 320" role="img" aria-label="Настоящая soma-проекция MaleCNS"><defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5z" fill="context-stroke"/></marker></defs><g>{''.join(background)}</g><g class="edge-layer">{''.join(links)}</g><g class="node-layer">{''.join(dots)}</g><text x="18" y="313" class="anatomy-label">BRAIN · annotated somata</text><text x="315" y="270" class="anatomy-label">VENTRAL NERVE CORD · model state</text></svg><div class="history-strip"><span>сила сигнала по последним решениям</span>{history_bars}</div></div>'''

def action_name(action):
    if action == TAKE: return "берёт"
    if action == PASS: return "пас"
    return RANK_NAMES[action % 9] + SUIT_NAMES[action // 9]

class Match:
    def __init__(self, graph, checkpoint, device, seed, human):
        self.device, self.body_ids = device, getattr(graph, "body_ids", None)
        self._prepare_anatomy(getattr(graph, "soma_positions", None), getattr(graph, "anatomy_regions", None))
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        klass = DeepFlyPolicy if saved.get("architecture") == "deep" else FlyPolicy
        self.model = klass(graph, 154, 38, plastic_edges=saved.get("plastic_edges", False)).to(device).eval()
        self.model.load_state_dict(saved["model"])
        self.reset(seed, human)

    def _prepare_anatomy(self, positions, regions):
        self.projected_positions, self.anatomy = None, []
        if positions is None: return
        xyz = positions.detach().float().cpu()
        valid = torch.isfinite(xyz).all(1)
        if not valid.any(): return
        # Frontal view used by connectome viewers: anteroposterior Y vs dorsoventral Z.
        region = torch.zeros(len(xyz), dtype=torch.uint8) if regions is None else regions.detach().cpu().to(torch.uint8)
        projected = torch.full((len(xyz), 2), float("nan"))
        # Brain and ventral nerve cord require independent orientations/scales.
        for code, axes, left, top, width, height in ((0, (1, 0), 18, 10, 245, 280), (1, (0, 2), 315, 60, 280, 180)):
            mask = valid & (region == code)
            if not mask.any(): continue
            view = xyz[:, axes]
            low = torch.quantile(view[mask], .01, dim=0); high = torch.quantile(view[mask], .99, dim=0)
            scaled = ((view - low) / (high - low).clamp_min(1)).clamp(0, 1)
            projected[mask, 0] = left + scaled[mask, 0] * width
            projected[mask, 1] = top + (1 - scaled[mask, 1]) * height
        self.projected_positions = projected
        valid_indices = valid.nonzero().flatten()
        stride = max(1, len(valid_indices) // 6500)
        self.anatomy = projected[valid_indices[::stride]].tolist()

    def reset(self, seed=None, human=None):
        self.seed = random.randrange(2**31) if seed is None else seed
        self.human = random.randrange(2) if human is None else human
        self.ai, self.game = 1 - self.human, DurakGame(self.seed)
        self.state = self.model.initial_state(1, self.device)
        self.last_ai, self.last_activity, self.last_edges, self.last_coordinates, self.last_choices, self.activity_history = [], [], [], {}, [], []
        self.advance()

    def _capture_thought(self, logits, mask):
        brain_size = getattr(self.model, "brain_size", getattr(self.model, "neuron_count", self.state.shape[-1]))
        brain = self.state[0, :brain_size]
        _, indices = torch.topk(brain.abs(), min(420, brain.numel()))
        signed, ids = brain[indices], indices
        if self.body_ids is not None: ids = self.body_ids.to(indices.device)[indices]
        self.last_activity = list(zip(ids.detach().cpu().tolist(), signed.detach().float().cpu().tolist()))
        if self.projected_positions is not None:
            coords = self.projected_positions[indices.detach().cpu()]
            self.last_coordinates.update({int(body): tuple(map(float, xy)) for body, xy in zip(ids.detach().cpu().tolist(), coords.tolist()) if math.isfinite(xy[0])})
        self.activity_history = (self.activity_history + [self.last_activity[:120]])[-8:]
        edge_indices = self.model.edge_indices
        sources, targets = edge_indices[1], edge_indices[0]
        edge_weights = self.model.edge_weights
        edge_signal = brain[sources] * edge_weights
        edge_values, selected_edges = torch.topk(edge_signal.abs(), min(700, edge_signal.numel()))
        selected_sources, selected_targets = sources[selected_edges], targets[selected_edges]
        selected_signal, selected_weights = edge_signal[selected_edges], edge_weights[selected_edges]
        if self.body_ids is not None:
            body_ids = self.body_ids.to(selected_sources.device)
            source_bodies, target_bodies = body_ids[selected_sources], body_ids[selected_targets]
        else:
            source_bodies, target_bodies = selected_sources, selected_targets
        if self.projected_positions is not None:
            for bodies, points in ((source_bodies.detach().cpu().tolist(), self.projected_positions[selected_sources.detach().cpu()].tolist()), (target_bodies.detach().cpu().tolist(), self.projected_positions[selected_targets.detach().cpu()].tolist())):
                self.last_coordinates.update({int(body): tuple(map(float, xy)) for body, xy in zip(bodies, points) if math.isfinite(xy[0])})
        self.last_edges = list(zip(
            source_bodies.detach().cpu().tolist(), target_bodies.detach().cpu().tolist(),
            selected_signal.detach().float().cpu().tolist(), selected_weights.detach().float().cpu().tolist(),
        ))
        probabilities = logits.masked_fill(~mask, -1e9).softmax(-1)[0]
        probs, actions = torch.topk(probabilities, min(4, int(mask.sum().item())))
        self.last_choices = list(zip(actions.cpu().tolist(), probs.cpu().tolist()))

    def advance(self):
        while self.game.s.winner is None and self.game.s.actor == self.ai:
            legal = self.game.legal(); mask = torch.zeros(1, 38, dtype=torch.bool, device=self.device); mask[0, legal] = True
            with torch.no_grad():
                logits, _, self.state = self.model(observation(self.game, self.ai, self.device), self.state)
                self._capture_thought(logits, mask)
            action = int(logits.masked_fill(~mask, -1e9).argmax(-1).item())
            self.last_ai = (self.last_ai + [action])[-3:]
            self.game.play(action)

    def play(self, action):
        if self.game.s.winner is None and self.game.s.actor == self.human and action in self.game.legal():
            self.game.play(action); self.advance()

def page(match):
    global _RENDER_COORDINATES, _RENDER_ANATOMY
    _RENDER_COORDINATES, _RENDER_ANATOMY = match.last_coordinates, match.anatomy
    g, s = match.game, match.game.s
    legal = set(g.legal()) if s.winner is None and s.actor == match.human else set()
    role = "АТАКУЕШЬ" if s.attacker == match.human else "ЗАЩИЩАЕШЬСЯ"
    if s.phase == "throw": role = "СОПЕРНИК БЕРЁТ — МОЖНО ПОДКИНУТЬ" if s.attacker == match.human else "ТЫ БЕРЁШЬ"
    mini_cards = "".join(
        f'<i style="--n:{index};--count:{len(s.hands[match.ai])}"></i>'
        for index, _ in enumerate(s.hands[match.ai])
    )
    fly_avatar = f'<div class="fly-avatar"><div class="fly-face">🪰<span class="scratch">〰</span></div><div class="fly-caption"><b>МУХА</b><span>{len(s.hands[match.ai])} карт</span></div><div class="mini-hand">{mini_cards}</div></div>'
    opponent = fly_avatar + "".join('<div class="card back">🪰</div>' for _ in s.hands[match.ai])
    table = "".join(f'<div class="pair">{card_html(a)}<div class="arrow">↓</div>{card_html(d) if d is not None else "<div class=empty>?</div>"}</div>' for a,d in s.table) or '<div class="empty-table">Стол пуст</div>'
    controls = (f'<button class="control take" name="action" value="{TAKE}">БЕРУ КАРТЫ</button>' if TAKE in legal else "") + (f'<button class="control pass" name="action" value="{PASS}">ЗАКОНЧИТЬ / ПАС</button>' if PASS in legal else "")
    # This inner wrapper keeps the compact page template's nested game column balanced.
    controls += '<div class="layout-balance">'
    if s.winner is not None: status = "НИЧЬЯ" if s.winner == -1 else "ТЫ ПОБЕДИЛ 🎉" if s.winner == match.human else "МУХА ПОБЕДИЛА 🪰"
    elif s.actor == match.human: status = role + " — выбери подсвеченную карту"
    else: status = "Муха думает…"
    last = ", ".join(action_name(a) for a in match.last_ai) or "—"
    choices = "".join(f'<div class="choice"><span>{html.escape(action_name(a))}</span><i style="width:{p*100:.1f}%"></i><b>{p:.0%}</b></div>' for a,p in match.last_choices) or '<div class="activity-empty">Решение ещё не принималось.</div>'
    neurons = ", ".join(f"{n} ({v:+.2f})" for n,v in match.last_activity[:6]) or "—"
    positive_edges = sum(signal >= 0 for _, _, signal, _ in match.last_edges)
    negative_edges = len(match.last_edges) - positive_edges
    max_activation = max((abs(value) for _, value in match.last_activity), default=0.0)
    metrics = f'<div class="metric"><b>{len(match.last_activity)}</b><span>активных узлов</span></div><div class="metric"><b>{len(match.last_edges)}</b><span>сильных рёбер</span></div><div class="metric hot"><b>{positive_edges}</b><span>возбуждающих</span></div><div class="metric cold"><b>{negative_edges}</b><span>подавляющих</span></div><div class="metric"><b>{max_activation:.3f}</b><span>peak |state|</span></div>'
    choices = f'<div class="monitor-toolbar"><button type="button" onclick="this.closest(\'.neural\').classList.toggle(\'hide-edges\')">связи</button><button type="button" onclick="this.closest(\'.neural\').classList.toggle(\'hide-nodes\')">нейроны</button><button type="button" onclick="this.closest(\'.neural\').requestFullscreen()">на весь экран</button></div><div class="metrics">{metrics}</div><div class="real-position-note">Позиции точек — настоящие <b>somaLocation</b> MaleCNS в EM-пространстве (8-нм voxels). Линии — реальные направленные рёбра.</div>' + choices
    restart = '<form method="post"><button class="control pass" name="action" value="new">НОВАЯ РАЗДАЧА</button><button class="control take" name="action" value="same">ПОВТОРИТЬ ЭТУ</button></form>' if s.winner is not None else ""
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Дурак против мухи</title><style>
#thinking{{display:none!important}}
.neural>.neural-note{{display:none}}.real-position-note{{font-size:11px;color:#83bbb2;border-left:2px solid #3fe0c2;padding:6px 8px;margin:7px 0}}
.card.unavailable{{opacity:.48;filter:saturate(.45);box-shadow:none}}.trump-group .card.unavailable{{border-color:#aaa}}button.legal::after{{content:'МОЖНО';position:absolute;top:-11px;background:#ffd54a;color:#152016;font-size:8px;padding:2px 5px;border-radius:5px}}button.card{{position:relative}}
.anatomy-label{{fill:#8abdb5;font-size:10px;letter-spacing:1px}}
.active-neuron{{animation:none!important}}.replaying .activity-link{{stroke-dasharray:3 22;animation:signal-travel .8s linear 2,edge-fade 1.65s ease-out forwards}}.replaying .active-neuron{{opacity:0;animation:neuron-arrival .42s ease-out forwards!important;animation-delay:var(--arrival,.6s)!important}}@keyframes signal-travel{{to{{stroke-dashoffset:-50}}}}@keyframes edge-fade{{0%{{filter:brightness(1.25);opacity:.9}}75%{{opacity:.55}}100%{{opacity:.18}}}}@keyframes neuron-arrival{{0%{{opacity:0}}100%{{opacity:1}}}}
.fly-avatar{{position:relative;display:grid;grid-template-columns:55px auto;grid-template-rows:auto auto;align-items:center;text-align:left;min-width:145px;padding:8px 12px;border-radius:14px;background:#071f15;border:1px solid #5fe0a433;overflow:visible}}.fly-face{{grid-row:1/3;font-size:38px;transform-origin:center}}.scratch{{position:absolute;left:40px;top:3px;font-size:22px;color:#ffd166;opacity:0}}.fly-caption{{display:flex;flex-direction:column}}.fly-caption span{{font-size:11px;color:#8fbba4}}.mini-hand{{display:flex;height:20px;align-items:end;padding-left:5px}}.mini-hand i{{display:block;width:12px;height:17px;margin-left:-5px;background:#eaf3ff;border:1px solid #8294b2;border-radius:2px;transform-origin:50% 120%;transform:rotate(calc((var(--n,0) - (var(--count,1) - 1)/2) * 3deg))}}.fly-avatar.thinking .fly-face{{animation:fly-think .46s ease-in-out infinite alternate}}.fly-avatar.thinking .scratch{{opacity:1;animation:scratch-head .32s ease-in-out infinite alternate}}.fly-avatar.throwing::after{{content:'🂠';position:absolute;right:12px;bottom:10px;font-size:25px;z-index:4;animation:fly-toss .9s cubic-bezier(.2,.8,.25,1) forwards}}@keyframes fly-think{{to{{transform:rotate(-7deg) translateY(-2px)}}}}@keyframes scratch-head{{to{{transform:translate(5px,5px) rotate(25deg)}}}}@keyframes fly-toss{{0%{{opacity:1;transform:translate(0,0) rotate(-12deg)}}100%{{opacity:0;transform:translate(-150px,160px) rotate(40deg)}}}}.player-flight{{animation:player-toss .2s ease-out}}@keyframes player-toss{{from{{transform:translateY(90px) scale(.88);opacity:.55}}to{{transform:none;opacity:1}}}}
body{{margin:0;background:radial-gradient(circle at 30% 0,#17613a,#071d15 75%);color:#fff;font-family:Inter,system-ui,sans-serif;text-align:center}}.wrap{{max-width:1800px;margin:auto;padding:18px}}h1{{margin:4px}}.status{{font-size:22px;font-weight:800;background:#0b2f1a;padding:13px;border-radius:14px;margin:10px}}.meta{{font-size:16px;margin:10px}}.zone{{background:#0e3b20;border:1px solid #ffffff12;border-radius:18px;padding:12px;margin:10px}}.cards,.table{{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;min-height:105px;align-items:center}}.hand-groups{{display:flex;align-items:flex-end;justify-content:center;gap:12px;flex-wrap:wrap}}.suit-group{{padding:9px;border-radius:14px;background:#092f19;border:2px solid #ffffff1c}}.suit-label{{font-size:16px;font-weight:900;margin-bottom:7px}}.trump-group{{border:3px solid #ffd166;background:#3b3512;box-shadow:0 0 18px #ffd16644}}.trump-group .suit-label{{color:#ffe394}}.card{{width:72px;height:99px;background:#fff;border:3px solid #ddd;border-radius:12px;display:inline-flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:0 4px 9px #0008;color:#111}}button.card{{cursor:pointer}}button.legal{{border:5px solid #ffd700;transform:translateY(-4px)}}button.legal:hover{{transform:translateY(-10px);box-shadow:0 10px 18px #000}}.red{{color:#d21f2b}}.rank{{font-size:27px;font-weight:900}}.suit{{font-size:30px;line-height:31px}}.name{{font-size:11px}}.back{{background:repeating-linear-gradient(45deg,#273c75,#273c75 8px,#192a56 8px,#192a56 16px);color:#fff;font-size:30px}}.pair{{display:flex;flex-direction:column;align-items:center}}.pair .card{{width:62px;height:82px}}.arrow{{font-size:15px}}.empty{{width:62px;height:82px;border:3px dashed #ffffff88;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:25px}}.empty-table{{font-size:22px;color:#ffffffaa}}.control{{font-size:18px;font-weight:800;padding:13px 22px;border:0;border-radius:12px;margin:7px;cursor:pointer}}.take{{background:#e67e22;color:#fff}}.pass{{background:#3498db;color:#fff}}.workspace{{display:grid;grid-template-columns:minmax(620px,1.2fr) minmax(540px,.8fr);gap:14px;align-items:start}}.game-column h2{{margin:5px}}.neural{{position:sticky;top:12px;background:linear-gradient(160deg,#06181b,#020b0d);border:1px solid #4ec9b055;text-align:left;box-shadow:0 18px 70px #0008;margin:10px 0;max-height:calc(100vh - 28px);overflow:auto}}.neural-head{{display:flex;justify-content:space-between;align-items:end;gap:12px;flex-wrap:wrap}}.neural h2{{letter-spacing:.10em;margin:2px 0;color:#d8fff7}}.neural-note{{color:#8abdb5;font-size:12px}}.neural-grid{{display:grid;grid-template-columns:1fr;gap:5px;align-items:center}}.brain-map{{display:block;width:100%;height:auto;min-height:310px;background:radial-gradient(circle at 45% 50%,#123b3d,#020b0d 70%);border-radius:12px}}.brain-outline{{fill:url(#tissue);stroke:#66ffdf66;stroke-width:2}}.nerve-cord{{fill:none;stroke:#66ffdf55;stroke-width:7;stroke-linecap:round}}.background-neuron{{fill:#8effe8;opacity:.23}}.activity-link{{fill:none}}.active-neuron{{animation:pulse 1.7s ease-in-out infinite;transform-box:fill-box;transform-origin:center}}@keyframes pulse{{50%{{opacity:.45;transform:scale(.72)}}}}.legend{{font-size:12px;color:#a8d7d0;margin-top:4px}}.hot{{color:#ffd166}}.cold{{color:#35d8ff}}.choice{{display:grid;grid-template-columns:80px 1fr 48px;gap:8px;align-items:center;margin:8px 0}}.choice i{{height:18px;background:linear-gradient(90deg,#19c6a3,#ffd166);border-radius:5px;min-width:2px}}.choice b{{text-align:right}}.thought-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.neurons{{font-size:12px;color:#9fcac4;line-height:1.5;overflow-wrap:anywhere}}.activity-empty{{color:#8abdb5;padding:35px}}.monitor-toolbar{{display:flex;gap:6px;margin:8px 0;flex-wrap:wrap}}.monitor-toolbar button{{border:1px solid #55d8c055;background:#0d2b2f;color:#bdf9ed;padding:6px 10px;border-radius:7px;cursor:pointer}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin:7px 0}}.metric{{padding:7px;background:#092226;border:1px solid #ffffff0e;border-radius:8px;text-align:center}}.metric b{{display:block;font-size:16px}}.metric span{{display:block;font-size:9px;color:#82aaa5;text-transform:uppercase}}.neural.hide-edges .edge-layer{{display:none}}.neural.hide-nodes .node-layer{{display:none}}.neural:fullscreen{{max-height:none;margin:0;border-radius:0;padding:24px}}.neural:fullscreen .brain-map{{height:65vh}}.history-strip{{height:30px;display:flex;align-items:end;gap:4px;padding:5px 8px;background:#031113;border-radius:0 0 10px 10px}}.history-strip span{{font-size:10px;color:#6fa79f;margin-right:auto;align-self:center}}.history-strip i{{display:block;width:11px;max-height:25px;background:#43dfc0;box-shadow:0 0 7px #43dfc0;border-radius:2px 2px 0 0}}#thinking{{position:fixed;inset:0;background:#020b0ddd;z-index:10;display:none;place-items:center;backdrop-filter:blur(5px)}}#thinking.on{{display:grid}}.thinking-card{{font-size:28px;font-weight:900;padding:35px 55px;border:1px solid #62ffe1;border-radius:20px;background:#061b1d;box-shadow:0 0 70px #35d8ff55}}.thinking-card span{{display:block;font-size:13px;color:#8abdb5;margin-top:8px}}small{{color:#cde7d5}}@media(max-width:1200px){{.workspace{{grid-template-columns:1fr}}.neural{{position:relative;top:auto;max-height:none}}}}@media(max-width:800px){{.card{{width:65px;height:92px}}.thought-grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style><body><div id="thinking"><div class="thinking-card">🪰 МУХА ДУМАЕТ<span>10,5 млн синаптических связей передают сигнал</span></div></div><div class="wrap" id="dashboard"><h1>🪰 Дурак против MaleCNS</h1><div class="status">{status}</div>{restart}<div class="meta">Козырь: <b>{SUIT_NAMES[s.trump]} {SUIT_WORDS[s.trump]}</b> · В колоде: <b>{len(s.deck)}</b> · Последние ходы мухи: {html.escape(last)}</div><main class="workspace"><div class="game-column"><div class="zone"><h2>Карты мухи: {len(s.hands[match.ai])}</h2><div class="cards">{opponent}</div></div><div class="zone"><h2>Стол</h2><div class="table">{table}</div></div><form method="post"><div class="zone"><h2>Твоя рука</h2><div class="hand-groups">{hand_html(s.hands[match.human], legal, s.trump)}</div><div>{controls}</div></div></div></form><small>Раздача seed={match.seed}</small></div><aside class="zone neural"><div class="neural-head"><div><h2>CNS / SIGNAL MONITOR</h2><div class="neural-note">Реальные сильнейшие сигналы connectome перед последним ходом</div></div><div class="legend"><span class="hot">● возбуждение</span> · <span class="cold">● подавление</span></div></div><div class="neural-grid"><div>{activity_svg(match.last_activity, match.last_edges, match.activity_history)}</div><div class="thought-grid"><div><h3>Варианты хода</h3>{choices}</div><div><h3>Самые активные нейроны</h3><div class="neurons">{html.escape(neurons)}</div></div></div></div><div class="neural-note">Линии — реальные направленные рёбра MaleCNS; яркость отражает |активация источника × вес ребра|. Позиции функциональные, поскольку XYZ/SWC-морфология не входит в checkpoint.</div></aside></main></div><script>document.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,button=e.submitter;const data=new URLSearchParams(new FormData(form));if(button&&button.name)data.set(button.name,button.value);const fly=document.querySelector('.fly-avatar');if(fly)fly.classList.add('thinking');if(button&&button.matches('button.card')){{const thrown=button.cloneNode(true);thrown.removeAttribute('name');thrown.removeAttribute('value');thrown.classList.remove('legal');thrown.classList.add('player-flight');button.style.visibility='hidden';const table=document.querySelector('.table');const defenseSlot=table.querySelector('.pair .empty');if(defenseSlot){{defenseSlot.replaceWith(thrown)}}else{{const emptyTable=table.querySelector('.empty-table');if(emptyTable)emptyTable.remove();const pair=document.createElement('div');pair.className='pair';pair.append(thrown);pair.insertAdjacentHTML('beforeend','<div class="arrow">↓</div><div class="empty">?</div>');table.append(pair)}}}}try{{const response=await fetch('/',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:data}});const next=new DOMParser().parseFromString(await response.text(),'text/html');const currentNeural=document.querySelector('.neural'),nextNeural=next.querySelector('.neural');const aiMoved=currentNeural&&nextNeural&&currentNeural.innerHTML!==nextNeural.innerHTML;if(aiMoved){{const preview=nextNeural.cloneNode(true);preview.classList.add('replaying');currentNeural.replaceWith(preview);const activeFly=document.querySelector('.fly-avatar');if(activeFly){{activeFly.classList.remove('thinking');activeFly.classList.add('throwing')}}await new Promise(resolve=>setTimeout(resolve,1200))}}document.querySelector('#dashboard').replaceWith(next.querySelector('#dashboard'));}}catch(error){{alert('Ошибка связи с мухой: '+error);location.reload()}}}});</script></body></html>'''

def main():
    p=argparse.ArgumentParser(); p.add_argument("--graph",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--host",default="0.0.0.0"); p.add_argument("--port",type=int,default=8765); p.add_argument("--seed",type=int); p.add_argument("--human-seat",type=int,choices=(0,1)); a=p.parse_args()
    device=torch.device(a.device); seed=a.seed if a.seed is not None else random.randrange(2**31); human=a.human_seat if a.human_seat is not None else random.randrange(2)
    match=Match(load_graph(a.graph,device),a.checkpoint,device,seed,human); app=FastAPI(title="Durak vs MaleCNS")
    @app.get("/",response_class=HTMLResponse)
    async def board(): return HTMLResponse(page(match))
    @app.post("/")
    async def move(request:Request):
        action=parse_qs((await request.body()).decode()).get("action",[-1])[0]
        if action=="new": match.reset()
        elif action=="same": match.reset(match.seed,match.human)
        else:
            try: match.play(int(action))
            except (ValueError,TypeError): pass
        return RedirectResponse("/",status_code=303)
    print(f"FastAPI: http://{a.host}:{a.port} seed={seed}, human-seat={human}",flush=True); uvicorn.run(app,host=a.host,port=a.port,workers=1)

if __name__=="__main__": main()
