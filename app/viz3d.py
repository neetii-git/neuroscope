"""WebGL (Three.js) point-cloud brain viewer for the Streamlit app.

Encodes a downsampled brain volume plus tumour labels / uncertainty as compact base64 typed arrays and
renders them client-side with additive-blended points. No server round-trips while the user interacts.
"""
from __future__ import annotations

import base64
import json
import io
import os
from typing import Optional

import numpy as np

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def _vendor_js() -> str:
    """Three.js r128 + OrbitControls inlined, so the viewer works with no internet connection."""
    out = []
    for f in ("three.min.js", "OrbitControls.js"):
        with open(os.path.join(_VENDOR, f), encoding="utf-8") as fh:
            out.append(fh.read())
    return "\n".join(out)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _norm(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nz = v[mask]
    lo, hi = (np.percentile(nz, [1, 99.5]) if nz.size else (0, 1))
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)


def _slice_png(sl: np.ndarray) -> str:
    from PIL import Image
    nz = sl[sl != 0]
    lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0, 1))
    g = (np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    a = np.where(sl != 0, 235, 0).astype(np.uint8)
    rgba = np.stack([g, g, g, a], -1)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def brain_html(vol: np.ndarray, labels: Optional[np.ndarray] = None, entropy: Optional[np.ndarray] = None,
               brain_step: int = 3, tumour_step: int = 2, height: int = 560, title: str = "", subtitle: str = "",
               slice_z: Optional[int] = None, autorotate: bool = True, show_controls: bool = True,
               volumes: Optional[dict] = None) -> str:
    """vol: (D,H,W) float, zero outside brain. labels: (D,H,W) int 0..3. entropy: (D,H,W) float."""
    vol = np.asarray(vol, np.float32)
    mask = vol != 0
    D, H, W = vol.shape

    # brain cloud
    bm = mask[::brain_step, ::brain_step, ::brain_step]
    idx = np.argwhere(bm).astype(np.int16) * brain_step
    inten = (_norm(vol, mask)[::brain_step, ::brain_step, ::brain_step][bm] * 255).astype(np.uint8)
    # keep the surface denser than the interior so the shape reads well
    rng = np.random.default_rng(0)
    keep = rng.random(len(idx)) < 0.55
    idx, inten = idx[keep], inten[keep]

    # tumour cloud
    t_pos = np.zeros((0, 3), np.int16); t_lab = np.zeros(0, np.uint8); t_ent = np.zeros(0, np.uint8)
    if labels is not None:
        lm = labels[::tumour_step, ::tumour_step, ::tumour_step]
        t_pos = (np.argwhere(lm > 0) * tumour_step).astype(np.int16)
        t_lab = lm[lm > 0].astype(np.uint8)
        if entropy is not None:
            e = entropy[::tumour_step, ::tumour_step, ::tumour_step][lm > 0]
            t_ent = (np.clip(e / float(np.log(2)), 0, 1) * 255).astype(np.uint8)
        else:
            t_ent = np.zeros(len(t_lab), np.uint8)

    slice_img = _slice_png(vol[slice_z]) if slice_z is not None else ""
    ctrl_display = "flex" if show_controls else "none"
    has_unc = "true" if (entropy is not None and len(t_ent)) else "false"

    return f"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600&display=swap');
 #ov h1{{font-family:'Space Grotesk','Inter',sans-serif}}
 html,body{{margin:0;height:100%;background:rgba(11,12,14,.42);overflow:hidden;font-family:'Inter','Segoe UI',system-ui,sans-serif}}
 #c{{width:100%;height:100%;display:block}}
 #ov{{position:absolute;left:22px;top:18px;color:#E5E7EB;pointer-events:none}}
 #ov h1{{margin:0;font-size:30px;font-weight:600;letter-spacing:-0.02em}}
 #ov p{{margin:6px 0 0 0;font-size:13px;color:#94A3B8;max-width:420px;line-height:1.4}}
 #bar{{position:absolute;right:14px;top:14px;display:{ctrl_display};gap:6px;flex-wrap:wrap;justify-content:flex-end;max-width:60%}}
 #bar button{{background:rgba(17,24,39,.75);color:#E5E7EB;border:1px solid #334155;border-radius:8px;padding:5px 10px;font-size:12px;cursor:pointer}}
 #bar button.on{{background:#F5C518;border-color:#F5C518;color:#0E0F11}}
 #leg{{position:absolute;left:22px;bottom:14px;color:#94A3B8;font-size:12px;display:flex;gap:14px}}
 #leg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
 .vol{{font-weight:500;color:#E5E7EB;margin-left:5px}} .tag{{position:absolute;transform:translate(-50%,-130%);background:rgba(11,18,32,.82);border:1px solid #334155;border-radius:7px;padding:3px 8px;font-size:11px;color:#E5E7EB;cursor:pointer;white-space:nowrap;pointer-events:auto}} .tag:hover{{border-color:#F5C518}}
 .lg{{cursor:pointer;user-select:none}} .lg:not(.on){{opacity:.35;text-decoration:line-through}} #cutw{{display:inline-flex;align-items:center;gap:6px}} #cut{{width:120px;accent-color:#F5C518}}
 #hint{{position:absolute;right:16px;bottom:12px;color:#64748B;font-size:11px}}
</style></head><body>
<canvas id="c"></canvas>
<div id="ov"><h1>{title}</h1><p>{subtitle}</p></div>
<div id="bar">
 <button id="bBrain" class="on">Brain</button>
 <button id="bTum" class="on">Tumour</button>
 <button id="bUnc">Uncertainty</button>
 <button id="bSlice" class="{'on' if slice_img else ''}" style="display:{'inline-block' if slice_img else 'none'}">Slice</button>
 <button id="bRot" class="{'on' if autorotate else ''}">Rotate</button>
 <button id="bFocus">Focus tumour</button>
 <button id="bReset">Reset</button>
 <button id="bShot">Screenshot</button>
</div>
<div id="leg"><span class="lg on" data-g="1"><i style="background:#ff3333"></i>necrotic core<b class="vol" id="v1"></b></span><span class="lg on" data-g="2"><i style="background:#33cc33"></i>edema<b class="vol" id="v2"></b></span><span class="lg on" data-g="3"><i style="background:#ffd91a"></i>enhancing<b class="vol" id="v3"></b></span><span id="cutw">cut through <input id="cut" type="range" min="0" max="100" value="100"></span><span id="legU" style="display:none"><i style="background:#F2F2F0"></i>uncertainty (bright = unsure)</span></div>
<div id="tags"></div>
<div id="hint">drag to rotate, scroll to zoom, right-drag to pan</div>
<script>{_vendor_js()}</script>
<script>
const b64=(s,T)=>{{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}};
const D={D},H={H},W={W};
const VOLS={json.dumps({str(k): float(v) for k, v in (volumes or {}).items()})};
const bp=b64("{_b64(idx)}",Int16Array), bi=b64("{_b64(inten)}",Uint8Array);
const tp=b64("{_b64(t_pos)}",Int16Array), tl=b64("{_b64(t_lab)}",Uint8Array), te=b64("{_b64(t_ent)}",Uint8Array);
const hasUnc={has_unc};
const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({{canvas,antialias:true,alpha:true,preserveDrawingBuffer:true}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
const scene=new THREE.Scene(); renderer.setClearColor(0x000000,0);
const cam=new THREE.PerspectiveCamera(38,1,1,5000);
const ctr=new THREE.OrbitControls(cam,canvas); ctr.enableDamping=true; ctr.dampingFactor=0.08; ctr.autoRotate={'true' if autorotate else 'false'}; ctr.autoRotateSpeed=0.9;
const cx=W/2,cy=H/2,cz=D/2, R=Math.max(D,H,W);
function resetCam(){{cam.position.set(cx+R*0.95,cy-R*1.1,cz+R*0.6);cam.up.set(0,0,1);ctr.target.set(cx,cy,cz);ctr.update();}}
resetCam();
function circleTex(){{const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d');const r=g.createRadialGradient(32,32,0,32,32,32);r.addColorStop(0,'rgba(255,255,255,1)');r.addColorStop(0.4,'rgba(255,255,255,.55)');r.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=r;g.fillRect(0,0,64,64);return new THREE.CanvasTexture(c);}}
const tex=circleTex();
// brain cloud: cool grey-blue, dimmer inside
const n=bp.length/3, pos=new Float32Array(n*3), col=new Float32Array(n*3);
const J={brain_step}*0.9; for(let i=0;i<n;i++){{pos[3*i]=bp[3*i+2]+(Math.random()-0.5)*J;pos[3*i+1]=bp[3*i+1]+(Math.random()-0.5)*J;pos[3*i+2]=bp[3*i]+(Math.random()-0.5)*J;const v=bi[i]/255;const k=0.18+0.55*v;col[3*i]=k*0.72;col[3*i+1]=k*0.86;col[3*i+2]=k*1.0;}}
const bg=new THREE.BufferGeometry(); bg.setAttribute('position',new THREE.BufferAttribute(pos,3)); bg.setAttribute('color',new THREE.BufferAttribute(col,3));
const brain=new THREE.Points(bg,new THREE.PointsMaterial({{size:{brain_step*1.6},map:tex,vertexColors:true,transparent:true,opacity:0.42,depthWrite:false,blending:THREE.AdditiveBlending,sizeAttenuation:true}}));
scene.add(brain);
// tumour cloud
const m=tp.length/3, tpos=new Float32Array(m*3), tcolL=new Float32Array(m*3), tcolU=new Float32Array(m*3);
const LC={{1:[1.0,0.2,0.2],2:[0.2,0.8,0.2],3:[1.0,0.85,0.1]}};
const JT={tumour_step}*0.8; for(let i=0;i<m;i++){{tpos[3*i]=tp[3*i+2]+(Math.random()-0.5)*JT;tpos[3*i+1]=tp[3*i+1]+(Math.random()-0.5)*JT;tpos[3*i+2]=tp[3*i]+(Math.random()-0.5)*JT;const c=LC[tl[i]]||[1,1,1];tcolL[3*i]=c[0];tcolL[3*i+1]=c[1];tcolL[3*i+2]=c[2];
 const u=te[i]/255; tcolU[3*i]=0.12+0.88*u; tcolU[3*i+1]=0.12+0.84*u; tcolU[3*i+2]=0.12+0.70*u;}}
const groups=[];
function mkGroup(labels,opacity,size,order,additive){{
  const idx=[];for(let i=0;i<m;i++)if(labels.indexOf(tl[i])>=0)idx.push(i);
  const P=new Float32Array(idx.length*3),CL=new Float32Array(idx.length*3),CU=new Float32Array(idx.length*3);
  idx.forEach((k,q)=>{{for(let a=0;a<3;a++){{P[3*q+a]=tpos[3*k+a];CL[3*q+a]=tcolL[3*k+a];CU[3*q+a]=tcolU[3*k+a];}}}});
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));g.setAttribute('color',new THREE.BufferAttribute(CL,3));
  const pm=new THREE.Points(g,new THREE.PointsMaterial({{size:size,map:tex,vertexColors:true,transparent:true,opacity:opacity,depthWrite:false,blending:additive?THREE.AdditiveBlending:THREE.NormalBlending,sizeAttenuation:true}}));
  pm.renderOrder=order;scene.add(pm);groups.push({{g:g,CL:CL,CU:CU,pm:pm,lab:labels[0]}});return pm;
}}
// edema (green) is a soft translucent shell; necrotic core (red) and enhancing (yellow) are drawn last and opaque so they stay visible
mkGroup([2],0.22,{tumour_step}*2.2,2,true);
mkGroup([3],0.6,{tumour_step}*2.4,3,false);
mkGroup([1],1.0,{tumour_step}*2.6,4,false);

// slice plane
let plane=null;
const sliceSrc="{slice_img}";
if(sliceSrc){{new THREE.TextureLoader().load(sliceSrc,t=>{{const g=new THREE.PlaneGeometry(W,H);const mt=new THREE.MeshBasicMaterial({{map:t,transparent:true,opacity:0.85,side:THREE.DoubleSide,depthWrite:false}});plane=new THREE.Mesh(g,mt);plane.position.set(cx,cy,{slice_z if slice_z is not None else 0});plane.rotation.x=0;plane.scale.y=-1;scene.add(plane);}});}}
// controls
const on=(id,f)=>document.getElementById(id).addEventListener('click',e=>{{f(e.currentTarget)}});
on('bBrain',b=>{{brain.visible=!brain.visible;b.classList.toggle('on',brain.visible)}});
let tumOn=true; on('bTum',b=>{{tumOn=!tumOn;applyVis();b.classList.toggle('on',tumOn)}});
const lgOn={{1:true,2:true,3:true}};
document.querySelectorAll('.lg').forEach(el=>el.addEventListener('click',()=>{{const k=+el.dataset.g;lgOn[k]=!lgOn[k];el.classList.toggle('on',lgOn[k]);applyVis();}}));
function applyVis(){{groups.forEach(o=>{{o.pm.visible=tumOn&&lgOn[o.lab];}});}}
renderer.localClippingEnabled=true;
const clip=new THREE.Plane(new THREE.Vector3(0,0,-1),1e6);
brain.material.clippingPlanes=[clip];groups.forEach(o=>{{o.pm.material.clippingPlanes=[clip];}});
document.getElementById('cut').addEventListener('input',e=>{{const f=e.target.value/100;clip.constant=f>=0.995?1e6:D*f;}});

let unc=false; on('bUnc',b=>{{if(!hasUnc)return;unc=!unc;groups.forEach(o=>{{o.g.setAttribute('color',new THREE.BufferAttribute(unc?o.CU:o.CL,3));o.g.attributes.color.needsUpdate=true;}});b.classList.toggle('on',unc);document.getElementById('legU').style.display=unc?'inline':'none';}});
on('bSlice',b=>{{if(plane){{plane.visible=!plane.visible;b.classList.toggle('on',plane.visible)}}}});
on('bRot',b=>{{ctr.autoRotate=!ctr.autoRotate;b.classList.toggle('on',ctr.autoRotate)}});
on('bReset',()=>resetCam());
if(!hasUnc)document.getElementById('bUnc').style.opacity=0.35;
// region centroids, tags, fly-to, screenshot
const REG={{1:{{n:'Necrotic core',c:[0,0,0],k:0}},2:{{n:'Edema',c:[0,0,0],k:0}},3:{{n:'Enhancing',c:[0,0,0],k:0}}}};
for(let i=0;i<m;i++){{const r=REG[tl[i]];if(!r)continue;r.c[0]+=tpos[3*i];r.c[1]+=tpos[3*i+1];r.c[2]+=tpos[3*i+2];r.k++;}}
Object.keys(REG).forEach(k=>{{const r=REG[k];if(r.k){{r.c=r.c.map(v=>v/r.k);}}const e=document.getElementById('v'+k);if(e&&VOLS[k]!==undefined)e.textContent=VOLS[k].toFixed(1)+' mL';}});
const tagsEl=document.getElementById('tags');
Object.keys(REG).forEach(k=>{{const r=REG[k];if(!r.k)return;const d=document.createElement('div');d.className='tag';d.textContent=r.n+(VOLS[k]!==undefined?' · '+VOLS[k].toFixed(1)+' mL':'');d.addEventListener('click',()=>flyTo(r.c,Math.max(40,Math.cbrt(r.k)*{tumour_step}*5)));r.el=d;tagsEl.appendChild(d);}});
let fly=null;
function flyTo(t,dist){{const dir=new THREE.Vector3().subVectors(cam.position,ctr.target).normalize();fly={{t0:performance.now(),from:cam.position.clone(),fromT:ctr.target.clone(),to:new THREE.Vector3(t[0],t[1],t[2]).add(dir.multiplyScalar(dist)),toT:new THREE.Vector3(t[0],t[1],t[2])}};ctr.autoRotate=false;document.getElementById('bRot').classList.remove('on');}}
function stepFly(){{if(!fly)return;const u=Math.min(1,(performance.now()-fly.t0)/900),e=u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;cam.position.lerpVectors(fly.from,fly.to,e);ctr.target.lerpVectors(fly.fromT,fly.toT,e);if(u>=1)fly=null;}}
function placeTags(){{const w=canvas.clientWidth,h=canvas.clientHeight;Object.keys(REG).forEach(k=>{{const r=REG[k];if(!r.el)return;const v=new THREE.Vector3(r.c[0],r.c[1],r.c[2]).project(cam);const vis=v.z<1&&tumOn&&lgOn[k];r.el.style.display=vis?'block':'none';r.el.style.left=((v.x*.5+.5)*w)+'px';r.el.style.top=((-v.y*.5+.5)*h+ (k==2?-26:(k==3?0:26)))+'px';}});}}
on('bFocus',()=>{{const r=REG[1].k?REG[1]:(REG[3].k?REG[3]:REG[2]);if(r.k)flyTo(r.c,Math.max(60,Math.cbrt(r.k)*{tumour_step}*6));}});
on('bShot',()=>{{renderer.render(scene,cam);const a=document.createElement('a');a.download='neuroscope_3d.png';a.href=renderer.domElement.toDataURL('image/png');a.click();}});

function resize(){{const w=canvas.clientWidth||window.innerWidth,h=canvas.clientHeight||window.innerHeight;renderer.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}}
window.addEventListener('resize',resize);
(function loop(){{resize();stepFly();ctr.update();placeTags();renderer.render(scene,cam);requestAnimationFrame(loop);}})();
</script></body></html>"""
