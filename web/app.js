const $ = (selector) => document.querySelector(selector);
const canvas = $('#arena');
const gl = canvas.getContext('webgl', {antialias: true, alpha: false});
const fallback = gl ? null : canvas.getContext('2d');
let state = null;
let selectedAgent = 'blue-1';
let orbit = {yaw: -0.15, pitch: 0.67, distance: 13.8};
let dragging = false;
let pointer = {x: 0, y: 0};
let toastTimer;

const colors = {
  field: [0.035, 0.25, 0.16, 1], fieldEdge: [0.025, 0.11, 0.09, 1],
  line: [0.72, 0.92, 0.85, 1], blue: [0.06, 0.68, 0.93, 1],
  orange: [1.0, 0.39, 0.10, 1], dark: [0.018, 0.028, 0.036, 1],
  wing: [0.58, 0.88, 0.93, .58], ball: [0.31, 0.13, 0.045, 1],
  ballLight: [0.48, 0.24, 0.09, 1], stand: [0.055, 0.075, 0.092, 1],
  light: [0.53, 0.67, 0.72, 1], shadow: [0.006, 0.012, 0.014, .38],
};

function showToast(message, error = false) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.style.borderColor = error ? '#833746' : '#294f64';
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
}

async function api(path, body) {
  const options = {method: 'POST', headers: {'Content-Type': 'application/json'}};
  if (body !== undefined) options.body = JSON.stringify(body);
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

// Small matrix library used by the dependency-free WebGL renderer.
const M = {
  identity: () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  multiply(a, b) {
    const out = new Array(16).fill(0);
    for (let c=0;c<4;c++) for (let r=0;r<4;r++) for (let k=0;k<4;k++) out[c*4+r] += a[k*4+r]*b[c*4+k];
    return out;
  },
  translation: (x,y,z) => [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1],
  scale: (x,y,z) => [x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1],
  rotateX(a) {const c=Math.cos(a),s=Math.sin(a);return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]},
  rotateY(a) {const c=Math.cos(a),s=Math.sin(a);return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]},
  rotateZ(a) {const c=Math.cos(a),s=Math.sin(a);return [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]},
  perspective(fov, aspect, near, far) {
    const f=1/Math.tan(fov/2), nf=1/(near-far);
    return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
  },
  lookAt(eye, center, up=[0,1,0]) {
    const norm=v=>{const l=Math.hypot(...v)||1;return v.map(n=>n/l)};
    const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    const z=norm([eye[0]-center[0],eye[1]-center[1],eye[2]-center[2]]),x=norm(cross(up,z)),y=cross(z,x);
    return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1];
  },
  model(x,y,z,sx,sy,sz,ry=0,rx=0,rz=0) {
    return M.multiply(M.translation(x,y,z),M.multiply(M.rotateY(ry),M.multiply(M.rotateX(rx),M.multiply(M.rotateZ(rz),M.scale(sx,sy,sz)))));
  }
};

function shader(type, source) {
  const value = gl.createShader(type); gl.shaderSource(value, source); gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
  return value;
}

let program, uniforms, attributes, meshes;
if (!gl) {
  $('#renderError').style.display = 'none';
} else {
  const vertex = shader(gl.VERTEX_SHADER, `
    attribute vec3 position; attribute vec3 normal;
    uniform mat4 model; uniform mat4 viewProjection;
    varying vec3 vNormal; varying vec3 vWorld;
    void main(){ vec4 world=model*vec4(position,1.0); vWorld=world.xyz; vNormal=normalize(mat3(model)*normal); gl_Position=viewProjection*world; }
  `);
  const fragment = shader(gl.FRAGMENT_SHADER, `
    precision mediump float; uniform vec4 color; uniform vec3 camera;
    varying vec3 vNormal; varying vec3 vWorld;
    void main(){
      vec3 light=normalize(vec3(-.35,.9,.4)); float diffuse=max(dot(normalize(vNormal),light),0.0);
      float rim=pow(1.0-max(dot(normalize(camera-vWorld),normalize(vNormal)),0.0),2.2);
      vec3 lit=color.rgb*(.34+diffuse*.76)+rim*color.rgb*.28;
      float fog=smoothstep(12.0,26.0,distance(camera,vWorld));
      gl_FragColor=vec4(mix(lit,vec3(.018,.03,.042),fog*.72),color.a);
    }
  `);
  program = gl.createProgram(); gl.attachShader(program,vertex); gl.attachShader(program,fragment); gl.linkProgram(program); gl.useProgram(program);
  attributes = {position:gl.getAttribLocation(program,'position'),normal:gl.getAttribLocation(program,'normal')};
  uniforms = {model:gl.getUniformLocation(program,'model'),viewProjection:gl.getUniformLocation(program,'viewProjection'),color:gl.getUniformLocation(program,'color'),camera:gl.getUniformLocation(program,'camera')};
  meshes = {cube:createCube(),sphere:createSphere(16,22),cylinder:createCylinder(18)};
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
}

function createMesh(positions, normals, indices) {
  const mesh={count:indices.length};
  mesh.position=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,mesh.position);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.STATIC_DRAW);
  mesh.normal=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,mesh.normal);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(normals),gl.STATIC_DRAW);
  mesh.index=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.index);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indices),gl.STATIC_DRAW);
  return mesh;
}

function createCube() {
  const p=[],n=[],i=[]; const faces=[[[1,0,0],[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]],[[-1,0,0],[-1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1]],[[0,1,0],[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],[[0,-1,0],[-1,-1,1],[-1,-1,-1],[1,-1,-1],[1,-1,1]],[[0,0,1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],[[0,0,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]];
  faces.forEach((f,fi)=>{for(let v=1;v<5;v++){p.push(...f[v]);n.push(...f[0])}const o=fi*4;i.push(o,o+1,o+2,o,o+2,o+3)}); return createMesh(p,n,i);
}

function createSphere(rows, cols) {
  const p=[],n=[],i=[]; for(let y=0;y<=rows;y++){const v=y/rows,phi=v*Math.PI;for(let x=0;x<=cols;x++){const u=x/cols,t=u*Math.PI*2;const nx=Math.sin(phi)*Math.cos(t),ny=Math.cos(phi),nz=Math.sin(phi)*Math.sin(t);p.push(nx,ny,nz);n.push(nx,ny,nz)}}
  for(let y=0;y<rows;y++)for(let x=0;x<cols;x++){const a=y*(cols+1)+x,b=a+cols+1;i.push(a,b,a+1,b,b+1,a+1)} return createMesh(p,n,i);
}

function createCylinder(sides) {
  const p=[],n=[],i=[]; for(let s=0;s<=sides;s++){const a=s/sides*Math.PI*2,x=Math.cos(a),z=Math.sin(a);p.push(x,-1,z,x,1,z);n.push(x,0,z,x,0,z)}
  for(let s=0;s<sides;s++){const a=s*2;i.push(a,a+1,a+2,a+1,a+3,a+2)}
  const start=p.length/3;p.push(0,-1,0,0,1,0);n.push(0,-1,0,0,1,0);for(let s=0;s<sides;s++){const a=s/sides*Math.PI*2,x=Math.cos(a),z=Math.sin(a);p.push(x,-1,z,x,1,z);n.push(0,-1,0,0,1,0)}for(let s=0;s<sides;s++){i.push(start,start+2+s,start+2+(s+1)%sides,start+1,start+2+(s+1)%sides,start+2+s)}return createMesh(p,n,i);
}

function draw(mesh, model, color) {
  gl.uniformMatrix4fv(uniforms.model,false,new Float32Array(model));gl.uniform4fv(uniforms.color,color);
  gl.bindBuffer(gl.ARRAY_BUFFER,mesh.position);gl.enableVertexAttribArray(attributes.position);gl.vertexAttribPointer(attributes.position,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,mesh.normal);gl.enableVertexAttribArray(attributes.normal);gl.vertexAttribPointer(attributes.normal,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.index);gl.drawElements(gl.TRIANGLES,mesh.count,gl.UNSIGNED_SHORT,0);
}

function worldPosition(x,y){return [(x-500)/80,(y-300)/80]}

function drawField() {
  draw(meshes.cube,M.model(0,-.15,0,6.5,.15,4),colors.fieldEdge);
  draw(meshes.cube,M.model(0,.005,0,6.25,.025,3.75),colors.field);
  const line=(x,z,sx,sz)=>draw(meshes.cube,M.model(x,.045,z,sx,.018,sz),colors.line);
  line(0,0,.018,3.68); line(0,-3.69,6.15,.018); line(0,3.69,6.15,.018); line(-6.14,0,.018,3.7); line(6.14,0,.018,3.7);
  // Centre circle made from small luminous segments.
  for(let a=0;a<Math.PI*2;a+=Math.PI/16) line(Math.cos(a)*1.05,Math.sin(a)*1.05,.13,.025);
  draw(meshes.cylinder,M.model(0,.07,0,.07,.02,.07),colors.line);
  // Goals: two posts and a crossbar on each end.
  for(const side of [-1,1]){
    const x=side*6.36; for(const z of [-1.3,1.3]) draw(meshes.cylinder,M.model(x,.58,z,.055,.58,.055),colors.line);
    draw(meshes.cylinder,M.model(x,1.13,0,.055,1.3,.055,0,Math.PI/2),colors.line);
    for(let z=-1.3;z<=1.31;z+=.26) draw(meshes.cylinder,M.model(x+side*.34,.48,z,.012,.42,.012),[.48,.62,.64,.35]);
  }
  // Stadium terraces and blue/orange light strips.
  for(const z of [-4.55,4.55]){
    draw(meshes.cube,M.model(0,-.05,z,7.5,.35,.55),colors.stand);
    draw(meshes.cube,M.model(0,.32,z,7.2,.055,.03),z<0?colors.blue:colors.orange);
  }
  for(const x of [-7.45,7.45]) draw(meshes.cube,M.model(x,-.03,0,.5,.3,4),colors.stand);
}

function drawFly(fly, time) {
  const [x,z]=worldPosition(fly.x,fly.y), angle=-fly.angle, teamColor=fly.team==='blue'?colors.blue:colors.orange;
  const flap=Math.sin(time*.018+fly.number*2.1)*.18;
  draw(meshes.sphere,M.model(x,.055,z,.34,.025,.23),colors.shadow);
  draw(meshes.sphere,M.model(x,.34,z,.30,.17,.15,angle),colors.dark);
  const dx=Math.cos(fly.angle),dz=Math.sin(fly.angle), px=-dz,pz=dx;
  draw(meshes.sphere,M.model(x+dx*.28,.37,z+dz*.28,.16,.15,.15,angle),teamColor);
  draw(meshes.sphere,M.model(x+px*.22-dx*.06,.48+flap,z+pz*.22-dz*.06,.28,.035,.16,angle+.42,0,-.22),colors.wing);
  draw(meshes.sphere,M.model(x-px*.22-dx*.06,.48-flap,z-pz*.22-dz*.06,.28,.035,.16,angle-.42,0,.22),colors.wing);
  // Bright eyes make player direction readable from the broadcast camera.
  draw(meshes.sphere,M.model(x+dx*.4+px*.07,.41,z+dz*.4+pz*.07,.035,.045,.035),[.75,1,.32,1]);
  draw(meshes.sphere,M.model(x+dx*.4-px*.07,.41,z+dz*.4-pz*.07,.035,.045,.035),[.75,1,.32,1]);
  if(state && state.lastTouch===fly.id) draw(meshes.cylinder,M.model(x,.025,z,.42,.012,.42),[teamColor[0],teamColor[1],teamColor[2],.48]);
}

function drawBall(time) {
  if(!state)return;const [x,z]=worldPosition(state.ball.x,state.ball.y);
  draw(meshes.sphere,M.model(x,.045,z,.25,.025,.25),colors.shadow);
  draw(meshes.sphere,M.model(x,.29,z,.21,.21,.21,time*.0005),colors.ball);
  for(let a=0;a<5;a++)draw(meshes.sphere,M.model(x+Math.cos(a*1.256)*.14,.33,z+Math.sin(a*1.256)*.14,.045,.025,.045),colors.ballLight);
}

function renderFallback(time) {
  const ctx=fallback,dpr=Math.min(devicePixelRatio,2),w=Math.floor(canvas.clientWidth*dpr),h=Math.floor(canvas.clientHeight*dpr);
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  const project=(x,y,lift=0)=>{const v=y/600,row=.54+.4*v;return [w/2+(x/1000-.5)*w*row,h*.15+v*h*.73-lift*(.45+.55*v)]};
  const path=points=>{ctx.beginPath();points.forEach((p,i)=>{const q=project(...p);i?ctx.lineTo(...q):ctx.moveTo(...q)});};
  const bg=ctx.createLinearGradient(0,0,0,h);bg.addColorStop(0,'#071521');bg.addColorStop(1,'#020507');ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
  // Stadium seating and luminous rails.
  path([[0,-55],[1000,-55],[1000,0],[0,0]]);ctx.fillStyle='#111d26';ctx.fill();
  path([[0,600],[1000,600],[1000,675],[0,675]]);ctx.fillStyle='#0b131a';ctx.fill();
  for(let x=20;x<1000;x+=38){const a=project(x,-48),b=project(x,-10);ctx.strokeStyle=x%76?'#173044':'#452819';ctx.lineWidth=3*dpr;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
  // Perspective pitch.
  path([[0,0],[1000,0],[1000,600],[0,600]]);const grass=ctx.createLinearGradient(0,h*.15,0,h*.9);grass.addColorStop(0,'#123e31');grass.addColorStop(1,'#0b281f');ctx.fillStyle=grass;ctx.fill();ctx.strokeStyle='#abddd080';ctx.lineWidth=2*dpr;ctx.stroke();
  for(let y=0;y<600;y+=75){path([[0,y],[1000,y],[1000,y+37.5],[0,y+37.5]]);ctx.fillStyle='#ffffff05';ctx.fill()}
  ctx.strokeStyle='#d5f5e8aa';ctx.lineWidth=1.5*dpr;
  path([[500,0],[500,600]]);ctx.stroke();
  path([[0,195],[105,195],[105,405],[0,405]]);ctx.stroke();path([[1000,195],[895,195],[895,405],[1000,405]]);ctx.stroke();
  ctx.beginPath();for(let a=0;a<=Math.PI*2+.1;a+=.12){const q=project(500+Math.cos(a)*85,300+Math.sin(a)*85);a?ctx.lineTo(...q):ctx.moveTo(...q)}ctx.stroke();
  const centre=project(500,300);ctx.fillStyle='#e4fff0';ctx.beginPath();ctx.arc(...centre,3*dpr,0,Math.PI*2);ctx.fill();
  // Goals sit above the field and provide a strong depth cue.
  const goal=(x,side)=>{const a=project(x,195),b=project(x,405),lift=44*dpr;ctx.strokeStyle='#e7fff4';ctx.lineWidth=4*dpr;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(a[0],a[1]-lift);ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0],b[1]-lift);ctx.moveTo(a[0],a[1]-lift);ctx.lineTo(b[0],b[1]-lift);ctx.stroke();ctx.strokeStyle='#8daaa755';ctx.lineWidth=1*dpr;for(let j=0;j<6;j++){const yy=195+j*42;const p=project(x,yy);ctx.beginPath();ctx.moveTo(p[0],p[1]-lift);ctx.lineTo(p[0]+side*25*dpr,p[1]-lift*.35);ctx.stroke()}};goal(0,-1);goal(1000,1);
  if(!state)return;
  const actors=[...state.flies.map(f=>({...f,kind:'fly'})),{...state.ball,kind:'ball',y:state.ball.y}].sort((a,b)=>a.y-b.y);
  actors.forEach(actor=>{
    const [x,y]=project(actor.x,actor.y),scale=dpr*(.7+actor.y/850);
    ctx.save();ctx.translate(x,y);
    ctx.fillStyle='#0008';ctx.beginPath();ctx.ellipse(0,5*scale,22*scale,7*scale,0,0,Math.PI*2);ctx.fill();
    if(actor.kind==='ball'){
      const r=12*scale,gradient=ctx.createRadialGradient(-r*.35,-r*.5,1,0,0,r);gradient.addColorStop(0,'#98602f');gradient.addColorStop(.45,'#603718');gradient.addColorStop(1,'#211007');ctx.fillStyle=gradient;ctx.beginPath();ctx.arc(0,-r*.55,r,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#b3753a';ctx.lineWidth=scale;ctx.stroke();
    }else{
      ctx.rotate(actor.angle);const flap=Math.sin(time*.02+actor.number)*.22;
      ctx.fillStyle='#bde9ec77';ctx.beginPath();ctx.ellipse(-5*scale,-10*scale,19*scale,6*scale,-.45-flap,0,Math.PI*2);ctx.ellipse(-5*scale,10*scale,19*scale,6*scale,.45+flap,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#d7ffff99';ctx.lineWidth=.7*scale;ctx.stroke();
      const color=actor.team==='blue'?'#18bfee':'#f56f22';const body=ctx.createLinearGradient(-15*scale,0,15*scale,0);body.addColorStop(0,'#071016');body.addColorStop(.7,'#152531');body.addColorStop(1,color);ctx.fillStyle=body;ctx.beginPath();ctx.ellipse(0,0,19*scale,9*scale,0,0,Math.PI*2);ctx.fill();ctx.fillStyle=color;ctx.beginPath();ctx.arc(17*scale,0,8*scale,0,Math.PI*2);ctx.fill();ctx.fillStyle='#caff51';ctx.beginPath();ctx.arc(22*scale,-3*scale,2.2*scale,0,Math.PI*2);ctx.arc(22*scale,3*scale,2.2*scale,0,Math.PI*2);ctx.fill();
    }ctx.restore();
  });
  // Soft broadcast glow.
  const glow=ctx.createRadialGradient(w*.5,h*.45,10,w*.5,h*.45,w*.55);glow.addColorStop(0,'#47dfff08');glow.addColorStop(1,'#00000055');ctx.fillStyle=glow;ctx.fillRect(0,0,w,h);
}

function render(time=0) {
  requestAnimationFrame(render); if(!gl){renderFallback(time);return}
  const dpr=Math.min(devicePixelRatio,2),width=Math.floor(canvas.clientWidth*dpr),height=Math.floor(canvas.clientHeight*dpr);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;gl.viewport(0,0,width,height)}
  gl.clearColor(.012,.024,.034,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const cp=Math.cos(orbit.pitch),eye=[Math.sin(orbit.yaw)*cp*orbit.distance,Math.sin(orbit.pitch)*orbit.distance,Math.cos(orbit.yaw)*cp*orbit.distance];
  const vp=M.multiply(M.perspective(.72,width/height,.1,60),M.lookAt(eye,[0,0,0]));
  gl.useProgram(program);gl.uniformMatrix4fv(uniforms.viewProjection,false,new Float32Array(vp));gl.uniform3fv(uniforms.camera,eye);
  drawField(); if(state){state.flies.forEach(fly=>drawFly(fly,time));drawBall(time)}
}

function formatReward(value){return `${value>=0?'+':''}${Number(value||0).toFixed(3)}`}
function updateUI(s) {
  state=s; $('#blueScore').textContent=s.score.blue;$('#orangeScore').textContent=s.score.orange;
  $('#episode').textContent=String(s.episode).padStart(3,'0');
  const seconds=Math.floor(s.step/60);$('#matchClock').textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
  $('#rounds').textContent=`${s.completedRounds} ROUND${s.completedRounds===1?'':'S'}`;
  $('#bluePossession').textContent=`${s.possessionPercent.blue}%`;$('#orangePossession').textContent=`${s.possessionPercent.orange}%`;$('#possessionBar').style.width=`${s.possessionPercent.blue}%`;
  $('#blueReward').textContent=formatReward(s.meanReward.blue);$('#orangeReward').textContent=formatReward(s.meanReward.orange);
  $('#sessionMode').textContent=s.paused?'MATCH PAUSED':s.training?'TRAINING LIVE':'EVALUATION LIVE';
  $('#liveDot').className=`live-dot ${s.paused?'paused':s.training?'':'eval'}`;
  $('#checkpointStatus').textContent=s.activeCheckpoint?s.activeCheckpoint.toUpperCase():(s.training?'UNSAVED POLICY':'CURRENT POLICY · FROZEN');
  $('#pauseButton').querySelector('b').textContent=s.paused?'Resume match':'Pause match';
  $('#modeButton').querySelector('b').textContent=s.training?'Evaluate policy':'Resume training';
  $('#teamSize').value=String(s.teamSize);
  const agentIds=s.flies.map(f=>f.id);
  if(!agentIds.includes(selectedAgent))selectedAgent=agentIds[0];
  const select=$('#agentSelect'); if(select.options.length!==agentIds.length){select.replaceChildren(...s.flies.map(f=>{const option=document.createElement('option');option.value=f.id;option.textContent=`${f.team==='blue'?'Azure':'Ember'} · Fly ${f.number}`;return option}))}select.value=selectedAgent;
  const values=s.activity[selectedAgent]||{forward:0,left:0,right:0,push:0};
  $('#actionBars').innerHTML=Object.entries(values).map(([name,value])=>`<div class="action-row"><span>${name}</span><div class="action-track"><i style="width:${Math.max(2,Math.min(100,(value+1)*50))}%"></i></div><b>${value>=0?'+':''}${value.toFixed(2)}</b></div>`).join('');
  const team=selectedAgent.split('-')[0],dopamine=s.dopamine[team]||0;
  $('#dopamineBar').style.width=`${Math.abs(dopamine)*100}%`;$('#dopamineBar').style.background=dopamine>=0?'var(--lime)':'var(--red)';$('#dopamineValue').textContent=formatReward(dopamine);
}

async function poll(){try{const response=await fetch('/api/state',{cache:'no-store'});updateUI(await response.json())}catch(error){$('#sessionMode').textContent='SERVER OFFLINE'}finally{setTimeout(poll,80)}}

async function refreshCheckpoints(){
  try{
    const checkpoints=await (await fetch('/api/checkpoints',{cache:'no-store'})).json(),list=$('#checkpointList');list.replaceChildren();
    if(!checkpoints.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No checkpoints yet';list.append(empty);return}
    checkpoints.forEach(item=>{const row=document.createElement('div');row.className='checkpoint-item';const info=document.createElement('div'),name=document.createElement('b'),meta=document.createElement('small'),button=document.createElement('button');name.textContent=`${item.active?'● ':''}${item.name}`;meta.textContent=`Round ${item.episode} · ${item.teamSize}v${item.teamSize}`;button.textContent='Evaluate';button.onclick=async()=>{try{await api('/api/checkpoint/load',{name:item.name,evaluate:true});showToast(`${item.name} loaded · learning frozen`);refreshCheckpoints()}catch(e){showToast(e.message,true)}};info.append(name,meta);row.append(info,button);list.append(row)});
  }catch(e){showToast('Could not read checkpoint vault',true)}
}

$('#pauseButton').onclick=()=>api('/api/pause').catch(e=>showToast(e.message,true));
$('#modeButton').onclick=()=>api('/api/mode').then(s=>showToast(s.training?'Training resumed':'Evaluation mode · learning frozen')).catch(e=>showToast(e.message,true));
$('#resetButton').onclick=()=>api('/api/reset').then(()=>{showToast('Brains reset to fresh weights');refreshCheckpoints()}).catch(e=>showToast(e.message,true));
$('#teamSize').onchange=e=>api('/api/team-size',{teamSize:Number(e.target.value)}).then(()=>showToast(`${e.target.value} vs ${e.target.value} match ready`)).catch(e=>showToast(e.message,true));
$('#agentSelect').onchange=e=>{selectedAgent=e.target.value};
$('#saveCheckpoint').onclick=async()=>{try{const result=await api('/api/checkpoint/save',{name:$('#checkpointName').value});showToast(`${result.name} saved`);refreshCheckpoints()}catch(e){showToast(e.message,true)}};
$('#checkpointUpload').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const data=JSON.parse(await file.text()),response=await fetch(`/api/checkpoint/import?name=${encodeURIComponent(file.name)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}),result=await response.json();if(!response.ok)throw new Error(result.error);showToast(`${result.name} imported`);refreshCheckpoints()}catch(e){showToast(e.message||'Invalid checkpoint file',true)}event.target.value=''};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target.tagName!=='INPUT'){e.preventDefault();$('#pauseButton').click()}});
canvas.addEventListener('pointerdown',e=>{dragging=true;pointer={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;orbit.yaw-=(e.clientX-pointer.x)*.007;orbit.pitch=Math.max(.25,Math.min(1.15,orbit.pitch+(e.clientY-pointer.y)*.005));pointer={x:e.clientX,y:e.clientY}});
canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',()=>dragging=false);
canvas.addEventListener('wheel',e=>{e.preventDefault();orbit.distance=Math.max(8.5,Math.min(22,orbit.distance+e.deltaY*.012))},{passive:false});

refreshCheckpoints();setInterval(refreshCheckpoints,5000);poll();render();
