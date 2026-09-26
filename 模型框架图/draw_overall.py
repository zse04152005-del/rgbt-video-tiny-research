from pathlib import Path
from html import escape
from math import atan2, cos, sin, pi

OUT = Path(__file__).with_name("01_Overall.svg")
W, H = 1900, 900
ink = "#253244"
rgb = "#e87668"
thermal = "#39a99f"
fused = "#4d8ed1"
control = "#8d6ac2"
gray = "#687586"

parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arr-r" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{rgb}"/></marker>
  <marker id="arr-t" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{thermal}"/></marker>
  <marker id="arr-f" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{fused}"/></marker>
  <marker id="arr-c" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{control}"/></marker>
  <marker id="arr-g" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{gray}"/></marker>
  <pattern id="grid" width="12" height="12" patternUnits="userSpaceOnUse"><path d="M12 0H0V12" fill="none" stroke="#bfd2e5" stroke-width="1"/></pattern>
</defs>
<rect width="100%" height="100%" fill="white"/>
<text x="950" y="54" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="31" font-weight="700" fill="{ink}">Reliability-Guided RGBT Video Tiny Object Detector</text>''']

def add(x): parts.append(x)
def rect(x,y,w,h,fill="white",stroke=ink,rx=12,dash="",sw=2):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def text(x,y,s,size=18,weight="500",anchor="middle",color=ink):
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')
def line(points,color=gray,width=3,dashed=False,arrow=True):
    d=" ".join(f'{x},{y}' for x,y in points)
    marker={rgb:"r",thermal:"t",fused:"f",control:"c",gray:"g"}.get(color,"g")
    add(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'+(' stroke-dasharray="8 6"' if dashed else '')+(f' marker-end="url(#arr-{marker})"' if arrow else '')+'/>')
    if arrow and len(points)>1:
        x0,y0=points[-2]; x1,y1=points[-1]
        a=atan2(y1-y0,x1-x0)
        p1=(x1-11*cos(a-.42),y1-11*sin(a-.42))
        p2=(x1-11*cos(a+.42),y1-11*sin(a+.42))
        add(f'<polygon points="{x1},{y1} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')
def slab(x,y,w,h,color,label=""):
    for off,opacity in [(16,.45),(8,.7),(0,1)]:
        add(f'<polygon points="{x+off},{y-off} {x+w+off},{y-off} {x+w+off},{y+h-off} {x+off},{y+h-off}" fill="{color}" fill-opacity="{.2*opacity}" stroke="{color}" stroke-width="1.6"/>')
    if label:text(x+w/2+8,y+h+26,label,17,"600")
def smallbox(x,y,w,h,label,fill="white",stroke=ink):
    rect(x,y,w,h,fill,stroke,9,"",1.6)
    text(x+w/2,y+h/2+6,label,17,"600")
def circle(x,y,symbol,color=ink,r=22):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="2.3"/>')
    text(x,y+8,symbol,27,"600",color=color)

# Input rows and independent encoders.
rect(40,153,142,114,"#fff7f5",rgb,4,"",1.8)
rect(40,370,142,114,"#f3fbfa",thermal,4,"",1.8)
for x,y,c in [(50,164,rgb),(50,381,thermal)]:
    add(f'<rect x="{x}" y="{y}" width="122" height="94" fill="{c}" fill-opacity=".11"/>')
    add(f'<path d="M{x} {y+62} L{x+40} {y+48} L{x+80} {y+55} L{x+122} {y+35}" fill="none" stroke="{c}" stroke-width="5" opacity=".7"/>')
    add(f'<rect x="{x+56}" y="{y+43}" width="12" height="12" fill="none" stroke="{c}" stroke-width="3"/>')
    add(f'<rect x="{x+91}" y="{y+33}" width="8" height="8" fill="none" stroke="{c}" stroke-width="3"/>')
text(111,144,"RGB_t",21,"700")
text(111,361,"Thermal_t",21,"700")
smallbox(224,174,128,72,"RGB Encoder","#fff0ed",rgb)
smallbox(224,391,128,72,"T Encoder","#edf9f8",thermal)
line([(182,210),(224,210)],rgb)
line([(182,427),(224,427)],thermal)
slab(390,180,65,75,rgb,"P2 P3 P4")
slab(390,397,65,75,thermal,"P2 P3 P4")
line([(352,210),(390,210)],rgb)
line([(352,427),(390,427)],thermal)

# Dual candidate seeds, explicitly both modalities.
rect(500,291,136,88,"#f7f8fb",gray,10)
text(568,319,"Dual Seeds",18,"700")
add(f'<circle cx="540" cy="347" r="7" fill="{rgb}"/><circle cx="584" cy="347" r="7" fill="{thermal}"/>')
line([(455,219),(480,219),(480,320),(500,320)],rgb)
line([(455,436),(480,436),(480,355),(500,355)],thermal)

# OCR: spatial correspondence and calibrated trust.
rect(680,144,348,370,"#f8f5fd",control,17,"9 6",2)
text(854,177,"OCR",24,"700",color=control)
smallbox(714,220,88,55,"FoV","white",control)
smallbox(822,220,88,55,"Search","white",control)
smallbox(930,220,70,55,"Corr.","white",control)
line([(802,248),(822,248)],control,2.4)
line([(910,248),(930,248)],control,2.4)
add('<rect x="741" y="303" width="118" height="95" fill="url(#grid)" stroke="#b6cce4" stroke-width="1.5"/>')
for x,y,c in [(778,340,rgb),(826,364,thermal),(788,375,thermal)]:add(f'<circle cx="{x}" cy="{y}" r="6" fill="{c}"/>')
smallbox(889,314,111,57,"r / valid","#f0e9fa",control)
text(800,428,"Local candidates",16,"500")
text(944,418,"Reliability",16,"500")
line([(455,202),(662,202),(662,235),(680,235)],rgb)
line([(455,419),(665,419),(665,360),(680,360)],thermal)
line([(636,335),(680,335)],gray)

# OCR outputs: selected thermal values and control.
slab(1062,323,48,64,thermal,"T candidates")
line([(1028,350),(1062,350)],thermal)

# TPSI: full RGB Q; sampled thermal KV; actual residual.
rect(1180,122,337,392,"#f2f8fd",fused,17,"9 6",2)
text(1349,157,"TPSI",24,"700",color=fused)
slab(1215,204,57,76,rgb,"Q full")
slab(1308,204,45,64,thermal,"K,V sparse")
smallbox(1390,222,101,63,"Cross-Attn","white",fused)
line([(1272,238),(1390,238)],rgb)
line([(1353,238),(1390,263)],thermal)
add('<rect x="1405" y="320" width="67" height="48" fill="url(#grid)" stroke="#9ac1e5" stroke-width="1.5"/>')
line([(1440,285),(1440,320)],fused)
line([(1440,368),(1440,414)],fused)
circle(1440,442,"+",fused,21)
slab(1472,404,29,65,fused)
text(1488,491,"F_t",19,"700")
line([(455,187),(468,187),(468,110),(1147,110),(1147,238),(1215,238)],rgb)
line([(1110,353),(1142,353),(1142,261),(1308,261)],thermal)
line([(1028,385),(1161,385),(1161,314),(1320,314)],control,2.2,True)
text(1245,336,"budget / gate",15,"500",color=control)
line([(1215,281),(1215,442),(1419,442)],rgb,2.4)

# Causal memory below; note prediction then update.
rect(991,635,158,84,"#f4f6f8",gray,10,"",1.6)
slab(1009,655,38,39,gray)
text(1100,674,"M past",18,"700")
text(1100,700,"t−1 … t−k",15,"500")
rect(1182,600,337,142,"#fff7f0","#e6a875",17,"9 6",2)
text(1348,629,"RGCM",23,"700",color="#b87545")
smallbox(1210,656,75,48,"Match","white","#d39d70")
smallbox(1303,656,75,48,"Gate","white","#d39d70")
circle(1425,680,"+",fused,21)
text(1492,686,"F_t*",20,"700",color=fused)
line([(1149,676),(1210,676)],gray)
line([(1285,680),(1303,680)],gray)
line([(1378,680),(1404,680)],gray)
line([(1446,680),(1470,680)],fused)
line([(1501,454),(1540,454),(1540,562),(1167,562),(1167,645),(1210,645)],fused)
line([(1501,454),(1560,454),(1560,755),(1425,755),(1425,701)],fused,2.6)
line([(1028,466),(1150,466),(1150,578),(1340,578),(1340,656)],control,2.2,True)
text(1250,572,"r / valid",15,"500",color=control)

# Head and detections in RGB coordinate frame.
smallbox(1575,639,143,82,"Detection Head","#f4f8fc",fused)
line([(1510,680),(1575,680)],fused)
line([(455,170),(455,83),(1646,83),(1646,639)],rgb,2.2)
text(895,79,"P3/P4 context",15,"500",color=rgb)
rect(1762,621,103,119,"#fff7f5",rgb,4,"",1.5)
for x,y,w,h in [(1778,668,20,21),(1813,647,13,17),(1833,682,17,16)]:add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{rgb}" stroke-width="2.5"/>')
text(1813,609,"Boxes_t",19,"700")
text(1813,763,"RGB coords",14,"500")
line([(1718,680),(1762,680)],fused)
# Memory update is delayed until after the head.
smallbox(1583,769,123,42,"Update M","#f6f7f8",gray)
line([(1646,721),(1646,769)],gray,2.2)
line([(1583,790),(1122,790),(1122,719)],gray,2.2)

# Consistent legend.
rect(41,826,1824,55,"#fbfcfd","#cbd3dd",10,"",1.5)
text(104,860,"LEGEND",17,"700")
line([(190,854),(240,854)],rgb,3);text(280,860,"RGB",16)
line([(338,854),(388,854)],thermal,3);text(435,860,"Thermal",16)
line([(508,854),(558,854)],fused,3);text(600,860,"Fused",16)
circle(707,854,"+",ink,14);text(762,860,"Add",16)
circle(852,854,"×",ink,14);text(907,860,"Gate",16)
circle(1001,854,"C",ink,14);text(1063,860,"Concat",16)
line([(1150,854),(1198,854)],control,2,True);text(1258,860,"Control",16)
slab(1381,842,28,26,gray);text(1478,860,"Feature map",16)
text(1703,860,"Past frames only",15,"500",color=gray)

add('</svg>')
OUT.write_text(''.join(parts), encoding='utf-8')
print(OUT)
