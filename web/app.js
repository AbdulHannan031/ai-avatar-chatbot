const form = document.getElementById('composer-form');
const input = document.getElementById('prompt');
const speakToggle = document.getElementById('speak-toggle');
const avatar = document.getElementById('avatar');
// chat container element (needed by addMessage)
const chatEl = document.getElementById('chat');

// --- 3D avatar player (loads `models/model.fbx` + animation FBX files) ---
// server serves the web folder at /static, so models live under /static/models/
const AVATAR_MODELS_PATH = '/static/models/';
let ANIM_FILES = []; // will be loaded from models/models.json at init
let GESTURE_MAP = {}; // optional mapping file models/gesture-map.json
let MODEL_OVERRIDES = {}; // optional global/per-file overrides loaded from models/model-overrides.json

const avatar3d = {
  ready: false,
  scene: null,
  camera: null,
  renderer: null,
  mixer: null,
  clock: null,
  root: null, // base skinned mesh/group
  actions: {}, // name -> AnimationAction
  currentAction: null,
  idleActionName: null,
  _gestureLoopTimer: null,
  _gestureLoopRunning: false,

  async init() {
    // try to load a local manifest of animation filenames for faster discovery
    try {
      const res = await fetch(AVATAR_MODELS_PATH + 'models.json');
      if (res.ok) {
        const arr = await res.json();
        if (Array.isArray(arr)) ANIM_FILES = arr;
      }
    } catch (e) {
      // ignore; fallback to static list later if needed
    }

    // try to load an optional gesture map (maps intent keys to exact clip names)
    try {
      const gm = await fetch(AVATAR_MODELS_PATH + 'gesture-map.json');
      if (gm.ok) GESTURE_MAP = await gm.json();
    } catch (e) {}

    // try to load optional model overrides (single place to tweak rotation/scale/zoom)
    try {
      const mo = await fetch(AVATAR_MODELS_PATH + 'model-overrides.json');
      if (mo.ok) MODEL_OVERRIDES = await mo.json();
    } catch (e) {}

    // load Three.js and FBXLoader as ES modules (avoid deprecated UMD builds)
    try {
      const THREE_MODULE_URL = 'https://unpkg.com/three@0.154.0/build/three.module.js';
      const FBX_LOADER_URL = 'https://unpkg.com/three@0.154.0/examples/jsm/loaders/FBXLoader.js';

  // import core three module
  const threeModule = await import(THREE_MODULE_URL);
  // Create a mutable copy of the module exports so we can attach additional loaders
  const THREE_NS = Object.assign({}, threeModule);
  // expose as global for existing code that references THREE
  window.THREE = THREE_NS;

      // Import FBXLoader directly as an ES module. The import map (in index.html)
      // maps the bare specifier 'three' to the three.module.js URL so internal
      // imports inside FBXLoader resolve correctly.
      try {
        const fbxModule = await import(FBX_LOADER_URL);
        // FBXLoader may be exported as a named export or default — handle both
        const FBXLoaderCtor = fbxModule.FBXLoader || fbxModule.default || fbxModule;
        // attach to global and THREE for backwards compatibility
        window.FBXLoader = FBXLoaderCtor;
        THREE.FBXLoader = FBXLoaderCtor;
      } catch (e) {
        // Fallback: try to fetch-and-rewrite as a last resort (previous approach)
        const resp = await fetch(FBX_LOADER_URL);
        if (!resp.ok) throw new Error('Failed to fetch FBXLoader module');
        let src = await resp.text();
        src = src.replace(/from\s+['\"]three['\"]/g, `from '${THREE_MODULE_URL}'`);
        src = src.replace(/from\s+['\"](\.\.?\/[^'\"]+)['\"]/g, (m, rel) => {
          try { return `from '${new URL(rel, FBX_LOADER_URL).href}'`; } catch (e) { return m; }
        });
        const blob = new Blob([src], { type: 'application/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        const fbxModule = await import(blobUrl);
        const FBXLoaderCtor = fbxModule.FBXLoader || fbxModule.default || fbxModule;
        window.FBXLoader = FBXLoaderCtor;
        THREE.FBXLoader = FBXLoaderCtor;
        URL.revokeObjectURL(blobUrl);
      }
    } catch (err) {
      console.error('Failed to load Three.js modules', err);
      try { showError('Failed to load 3D engine. Check network or allowlist scripts.'); } catch (e) {}
      throw err;
    }

  console.log('avatar3d.init: starting');
  // create canvas for 3D preview (do not attach to header avatar; keep header image visible)
    const canvas = document.createElement('canvas');
    canvas.id = 'avatar-canvas';
    // keep the header avatar image visible (we won't hide it)
    const img = document.getElementById('avatar-img');
    // loading overlay attached to the header avatar for a small loading hint
    const overlay = document.createElement('div'); overlay.className = 'avatar-overlay';
    overlay.innerHTML = '<div class="spinner"></div><div class="overlay-text">Loading avatar…</div>';
    avatar.appendChild(overlay);

    // By default show the right-side preview panel and move the canvas there so
    // the chatbot layout displays chat on the left and the avatar on the right
    try {
      openSidePreview();
      const previewBody = document.getElementById('preview-side-body');
    if (previewBody) previewBody.appendChild(canvas);
    // ensure renderer size matches preview panel and frame model using stored zoom
    try {
          updateRendererSize(this.renderer, canvas);
          const stored = parseFloat(localStorage.getItem('preview_zoom')) || 2.1;
          this.frameModel({ zoom: stored });
    } catch (e) {}
    } catch (e) {
      // ignore if preview-side not present yet
    }

  this.renderer = new window.THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    // make the clear color visible so we can tell if rendering occurs
    try { this.renderer.setClearColor(0x111315, 1); } catch (e) {}
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // set initial renderer size to match avatar container
    updateRendererSize(this.renderer, canvas);
    // watch for resizes
    window.addEventListener('resize', () => updateRendererSize(this.renderer, canvas));

  // make canvas visually obvious during debugging
  canvas.style.outline = '1px solid rgba(124, 252, 0, 0.12)';
  canvas.style.background = '#0b0c0d';
  // allow clicking the avatar to preview gestures
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', () => showGesturePreview(canvas));

  this.scene = new window.THREE.Scene();
  this.camera = new window.THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    this.camera.position.set(0, 1.6, 2.6);

  const hemi = new window.THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
    this.scene.add(hemi);
  const dir = new window.THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 7.5);
    this.scene.add(dir);

  this.clock = new window.THREE.Clock();

  // load base model first (blocking) so UI shows the avatar quickly
  await this.loadBaseModel();
  console.log('avatar3d.init: base model loaded, scene root:', this.root);
  // frame the model in the current camera/view
  try { this.frameModel(); } catch (e) { console.warn('frameModel failed', e); }

    // ensure ANIM_FILES fallback if manifest missing
    if (!ANIM_FILES || ANIM_FILES.length === 0) {
      ANIM_FILES = [
        'Talking.fbx','acknowledging.fbx','angry gesture.fbx','annoyed head shake.fbx','being cocky.fbx',
        'dismissing gesture.fbx','happy hand gesture.fbx','hard head nod.fbx','head nod yes.fbx','lengthy head nod.fbx',
        'look away gesture.fbx','relieved sigh.fbx','sarcastic head nod.fbx','shaking head no.fbx','Thoughtful Head Nod.fbx',
        'thoughtful head shake.fbx','weight shift.fbx'
      ];
    }

    // prioritized loading: ensure 'Talking' animation is loaded first for TTS sync
    const talkingFile = findAnimByName('talking') || 'Talking.fbx';
    try {
      await this.loadAnimationFile(talkingFile);
      const talkKey = normalizeKey('Talking');
      if (this.actions[talkKey]) {
        console.log('Playing Talking action for verification');
        this.play('Talking', { fade: 0.2 });
      } else {
        console.log('Talking action not found after load; registered actions:', Object.keys(this.actions));
      }
    } catch (e) { console.warn('Talking load failed', e); }

    // prefetch remaining animations in background (non-blocking)
    setTimeout(() => this.prefetchAnimations(), 300);

    // remove overlay once base model is present
    const ov = avatar.querySelector('.avatar-overlay'); if (ov) ov.remove();
    this.ready = true;
    this.animate();
  // start random gesture loop
  try { this.startGestureLoop(); } catch (e) {}
    // persist that avatar initialized so subsequent page loads can show UI immediately
    try { localStorage.setItem('avatar_initialized', '1'); } catch (e) {}
  },

  async loadBaseModel() {
    return new Promise((resolve, reject) => {
  const loader = new window.THREE.FBXLoader();
    loader.load(AVATAR_MODELS_PATH + 'model.fbx', (obj) => {
      // common fixes
      // Don't force a -90deg rotation for all models — instead detect orientation
      // We'll test a small set of candidate rotations and pick the one that
      // yields the largest Y dimension in the model's bounding box. This is a
      // robust heuristic to choose an 'upright' orientation across exporters.
      obj.rotation.x = 0;
      // add to scene first so Box3 can compute sizes
      this.root = obj;
      this.scene.add(obj);
      // compute bounding box and auto-scale model to a sensible size if needed
      try {
        const candidates = [0, -Math.PI / 2, Math.PI / 2, Math.PI];
        let best = { rot: 0, sizeY: 0, centerY: 0 };
        const box = new window.THREE.Box3();
        for (const r of candidates) {
          obj.rotation.x = r;
          // ensure world matrix applies
          obj.updateMatrixWorld(true);
          try {
            box.setFromObject(obj);
            const s = box.getSize(new window.THREE.Vector3());
            const c = box.getCenter(new window.THREE.Vector3());
            // prefer larger Y size; tie-break with higher center Y
            if (s.y > best.sizeY || (Math.abs(s.y - best.sizeY) < 1e-6 && c.y > best.centerY)) {
              best = { rot: r, sizeY: s.y, centerY: c.y };
            }
          } catch (e) {
            // ignore and continue
          }
        }
        // apply the best rotation found
        obj.rotation.x = best.rot;
        if (best.rot !== 0) console.log('loadBaseModel: auto-rotated model by', (best.rot * 180 / Math.PI).toFixed(0), 'degrees for best Y size');
        // recompute bounding box after final rotation for scaling
        box.setFromObject(obj);
        // apply any global overrides (per-file or default)
        try {
          const overrides = MODEL_OVERRIDES && (MODEL_OVERRIDES['model.fbx'] || MODEL_OVERRIDES['default']);
          if (overrides) {
            if (typeof overrides.rotX === 'number') obj.rotation.x = overrides.rotX;
            if (typeof overrides.scale === 'number') obj.scale.setScalar(overrides.scale);
            if (overrides.translate && Array.isArray(overrides.translate) && overrides.translate.length === 3) {
              obj.position.set(overrides.translate[0], overrides.translate[1], overrides.translate[2]);
            }
            if (overrides.log) console.log('Applied model override to base model', overrides);
          }
        } catch (e) { console.warn('Applying model overrides failed', e); }
        // Ensure model is initially front-facing: reset yaw/roll so preview shows
        // the character facing the camera. This prevents models that were
        // exported rotated around Y/Z from appearing sideways or back-faced.
        try {
          obj.rotation.y = 0;
          obj.rotation.z = 0;
          obj.updateMatrixWorld(true);
        } catch (e) {}
        const size = box.getSize(new window.THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        // If model is very small or very large, normalize so max dimension ~ 1.0
        if (maxDim > 0 && (maxDim < 0.2 || maxDim > 5)) {
            const target = 1.0; // desired approximate max dimension in scene units
            const scaleFactor = target / maxDim;
            // cap extreme scaling
            const safe = Math.max(0.01, Math.min(scaleFactor, 100));
            obj.scale.setScalar(safe);
            console.log('loadBaseModel: auto-scaled model by', safe, 'original maxDim', maxDim);
          }
        } catch (e) {
          console.warn('loadBaseModel: bounding box/scale failed', e);
        }
  this.mixer = new window.THREE.AnimationMixer(obj);
        resolve();
      }, undefined, (err) => reject(err));
    });
  },

  async loadGLBModel(path) {
    return new Promise((resolve, reject) => {
      if (typeof THREE.GLTFLoader === 'undefined') return reject(new Error('GLTFLoader not present'));
  const loader = new window.THREE.GLTFLoader();
      loader.load(path, (gltf) => {
        // remove fallback img
        const img = document.getElementById('avatar-img'); if (img) img.style.display = 'none';
        // adjust scale/rotation if needed
        gltf.scene.rotation.x = -Math.PI / 2;
        this.root = gltf.scene;
        this.scene.add(gltf.scene);
        try {
          const box = new window.THREE.Box3().setFromObject(gltf.scene);
          const size = box.getSize(new window.THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          if (maxDim > 0 && (maxDim < 0.2 || maxDim > 5)) {
            const target = 1.0;
            const scaleFactor = target / maxDim;
            const safe = Math.max(0.01, Math.min(scaleFactor, 100));
            gltf.scene.scale.setScalar(safe);
            console.log('loadGLBModel: auto-scaled gltf by', safe, 'original maxDim', maxDim);
          }
        } catch (e) {
          console.warn('loadGLBModel: bounding box/scale failed', e);
        }
  this.mixer = new window.THREE.AnimationMixer(gltf.scene);
        // register animations
        if (gltf.animations && gltf.animations.length) {
          for (const clip of gltf.animations) {
            const name = normalizeKey(clip.name || clip.uuid || 'clip');
            clip.name = name;
            this.actions[name] = this.mixer.clipAction(clip);
          }
        }
        resolve(true);
      }, (xhr) => {
        // progress ok
      }, (err) => {
        reject(err);
      });
    });
  },

  async loadAnimations() {
  const loader = new window.THREE.FBXLoader();
    for (const f of ANIM_FILES) {
      try {
        // eslint-disable-next-line no-await-in-loop
        const animObj = await new Promise((resolve, reject) => loader.load(AVATAR_MODELS_PATH + f, resolve, undefined, reject));
        if (animObj && animObj.animations && animObj.animations.length && this.mixer) {
          for (const clip of animObj.animations) {
            // create a normalized key from filename
            const key = normalizeKey(f.replace(/\.fbx$/i, ''));
            // ensure unique names if multiple clips present
            const name = clip.name && clip.name !== 'take_001' ? normalizeKey(clip.name) : key;
            clip.name = name;
            this.actions[name] = this.mixer.clipAction(clip);
          }
        }
      } catch (e) {
        // silently continue for any missing/bad animations
        // console.warn('anim load failed', f, e);
      }
    }
  },

  async loadAnimationFile(filename) {
    if (!filename) return;
    const key = normalizeKey(filename.replace(/\.fbx$/i, ''));
    if (this.actions[key]) return; // already loaded
  const loader = new window.THREE.FBXLoader();
    return new Promise((resolve, reject) => {
      loader.load(AVATAR_MODELS_PATH + filename, (animObj) => {
        if (animObj && animObj.animations && animObj.animations.length && this.mixer) {
          console.log('loadAnimationFile:', filename, 'clips:', animObj.animations.map(c=>c.name));
          for (const clip of animObj.animations) {
            // sanitize clip: remove root-motion tracks that translate/rotate the
            // top-level root bone (these often come from Mixamo/FBX and cause the
            // whole character to be laid down or displaced when played).
            const rawName = clip.name || key;
            const name = rawName && rawName !== 'take_001' ? normalizeKey(rawName) : key;
            // helper: determine if a track likely targets the root/hips/pelvis
            const isRootMotionTrack = (trackName) => {
              try {
                const base = String(trackName).split(/[.\/\/:#]/)[0].toLowerCase();
                const rootKeys = ['root','hips','pelvis','mixamo','mixamorig','bip','reference'];
                for (const rk of rootKeys) if (base.includes(rk)) return true;
                return false;
              } catch (e) { return false; }
            };

            // filter tracks: keep everything except position/quaternion/rotation on root bones
            let tracks = clip.tracks ? clip.tracks.filter(t => {
              const tn = t.name || '';
              const isTransform = /\.position$|\.quaternion$|\.rotation$/i.test(tn);
              if (!isTransform) return true;
              if (!isRootMotionTrack(tn)) return true;
              // otherwise drop this root-motion track
              return false;
            }) : [];

            // if tracks were removed, log for debugging
            if (tracks.length !== (clip.tracks ? clip.tracks.length : 0)) {
              console.log('loadAnimationFile: stripped root-motion tracks from', filename, 'clip', rawName);
            }

            // create a new sanitized clip instance
            const safeClip = new window.THREE.AnimationClip(name, clip.duration, tracks);
            const action = this.mixer.clipAction(safeClip);
            // ensure action is enabled and has a visible weight
            try {
              action.enabled = true;
              action.setEffectiveWeight(1);
              action.clampWhenFinished = true;
            } catch (e) {}
            this.actions[name] = action;
            // also index by filename-based key so filenames like 'Talking.fbx' still
            // provide a predictable 'talking' action even when internal clip names differ
            if (key && !this.actions[key]) this.actions[key] = action;
            // set idle action if we don't have one yet
            if (!this.idleActionName) {
              const candidates = ['idle','idle breathing','breathing','weight shift','talking'];
              for (const c of candidates) {
                if (this.actions[normalizeKey(c)]) { this.idleActionName = normalizeKey(c); break; }
              }
              if (!this.idleActionName) this.idleActionName = name; // first-available
            }
          }
          console.log('Registered actions:', Object.keys(this.actions));
          // update UI controls in preview so you can manually trigger animations
          try { updatePreviewControls(); } catch (e) { console.warn('updatePreviewControls failed', e); }
        }
        resolve(true);
      }, undefined, (err) => reject(err));
    });
  },

  prefetchAnimations() {
    const remaining = ANIM_FILES.filter(f => {
      const k = normalizeKey(f.replace(/\.fbx$/i, ''));
      return !this.actions[k];
    });
    // staggered loading to avoid spikes
    let delay = 0;
    for (const f of remaining) {
      setTimeout(() => {
        this.loadAnimationFile(f).catch(() => {});
      }, delay);
      delay += 250; // 250ms between each
    }
  },

  // Play a named action. loop can be 'repeat' (default), 'once', or a numeric THREE constant.
  play(name, { fade = 0.25, loop = 'repeat' } = {}) {
    if (!this.ready) return;
    const key = normalizeKey(name);
    const action = this.actions[key];
    if (!action) return;
    // determine loop mode
    let loopMode = (window.THREE && window.THREE.LoopRepeat) ? window.THREE.LoopRepeat : 2200;
    if (typeof loop === 'string') {
      if (loop === 'once') loopMode = (window.THREE && window.THREE.LoopOnce) ? window.THREE.LoopOnce : 2201;
      else loopMode = (window.THREE && window.THREE.LoopRepeat) ? window.THREE.LoopRepeat : 2200;
    } else if (typeof loop === 'number') {
      loopMode = loop;
    }

    try {
      action.reset();
      action.setLoop(loopMode, Infinity);
      action.fadeIn(fade);
      action.play();
      action.enabled = true;
    } catch (e) { console.warn('action play failed', e); }

    // Non-looping actions are left to complete; the gesture loop will schedule next gestures.

    if (this.currentAction && this.currentAction !== action) {
      try { this.currentAction.fadeOut(fade); } catch (e) {}
    }
    this.currentAction = action;
  },

  stopCurrent(fade = 0.2) {
    if (!this.currentAction) return;
    this.currentAction.fadeOut(fade);
    this.currentAction = null;
  },

  playIdle() {
    if (!this.ready) return;
    const idleKey = this.idleActionName;
    if (!idleKey) return;
    try {
      this.play(idleKey, { fade: 0.25, loop: 'repeat' });
    } catch (e) { console.warn('playIdle failed', e); }
  },

  // Start/stop a background loop that randomly plays gestures
  startGestureLoop(minInterval = 2500, maxInterval = 6000) {
    if (this._gestureLoopRunning) return;
    this._gestureLoopRunning = true;
    const loopOnce = async () => {
      if (!this._gestureLoopRunning) return;
      // don't interrupt talking or when TTS is active
      if (document.getElementById('speak-toggle') && document.getElementById('speak-toggle').checked) {
        // if speechSynthesis is speaking, wait and retry
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
          this._gestureLoopTimer = setTimeout(loopOnce, 1000);
          return;
        }
      }
      // gather candidate gestures (exclude idle and talking)
      const keys = Object.keys(this.actions || {}).filter(k => {
        if (!k) return false;
        const kk = k.toLowerCase();
        if (kk === (this.idleActionName || '').toLowerCase()) return false;
        if (kk.includes('talk') || kk.includes('speaking')) return false;
        return true;
      });
      if (!keys.length) {
        // no candidates; try again later
        this._gestureLoopTimer = setTimeout(loopOnce, Math.floor((minInterval + maxInterval) / 2));
        return;
      }
      // pick a random gesture
      const key = keys[Math.floor(Math.random() * keys.length)];
      const action = this.actions[key];
      if (!action) {
        this._gestureLoopTimer = setTimeout(loopOnce, minInterval);
        return;
      }
      try {
        // play once
        this.play(key, { fade: 0.12, loop: 'once' });
      } catch (e) {
        console.warn('gesture play failed', e);
      }
      // duration of clip (fallback to 1s)
      let dur = 1000;
      try {
        const clip = action.getClip ? action.getClip() : (action._clip || action.getRoot && action.getRoot().clip);
        if (clip && clip.duration) dur = Math.max(300, Math.floor(clip.duration * 1000));
      } catch (e) {}
      // schedule next gesture immediately after clip finished (small buffer) so gestures play back-to-back
      const buffer = 80; // ms to allow crossfade
      this._gestureLoopTimer = setTimeout(() => {
        // schedule next gesture immediately (no idle gap)
        this._gestureLoopTimer = setTimeout(loopOnce, buffer);
      }, dur + buffer);
    };
    // initial kick-off
    this._gestureLoopTimer = setTimeout(loopOnce, Math.floor(Math.random() * (maxInterval - minInterval) + minInterval));
  },

  stopGestureLoop() {
    this._gestureLoopRunning = false;
    if (this._gestureLoopTimer) {
      clearTimeout(this._gestureLoopTimer);
      this._gestureLoopTimer = null;
    }
  },

  // choose a gesture based on text heuristics
  gestureForText(text) {
    if (!text) return 'Talking';
    const t = text.toLowerCase();
    // if gesture map contains a matching rule, use it
    for (const k of Object.keys(GESTURE_MAP || {})) {
      try {
        const re = new RegExp(k, 'i');
        if (re.test(text)) return GESTURE_MAP[k];
      } catch (e) {
        // ignore invalid regex keys
      }
    }
    if (/\b(no|not|never|don't|dont)\b/.test(t)) return 'shaking head no';
    if (/\b(yes|sure|right|correct|indeed)\b/.test(t)) return 'head nod yes';
    if (/\b(thank|thanks|great|love|good)\b/.test(t)) return 'happy hand gesture';
    if (/\b(think|hmm|maybe|consider)\b/.test(t)) return 'Thoughtful Head Nod';
    if (/\b(angry|frustrat|annoy)\b/.test(t)) return 'angry gesture';
    // default speaking animation
    return 'Talking';
  },

  animate() {
    requestAnimationFrame(() => this.animate());
    const dt = this.clock.getDelta();
    if (this.mixer) this.mixer.update(dt);
    if (this.renderer && this.camera) {
      // render to the small square sized by CSS - keep camera focused
      this.renderer.render(this.scene, this.camera);
    }
  }
};

// --- TTS voice selection: prefer a male-sounding voice when available ---
let SELECTED_VOICE_NAME = null;
let SELECTED_VOICE = null;

function pickPreferredMaleVoice() {
  if (!('speechSynthesis' in window)) return null;
  const voices = window.speechSynthesis.getVoices() || [];
  if (!voices.length) return null;
  // load stored preference first
  try { SELECTED_VOICE_NAME = localStorage.getItem('tts_voice'); } catch (e) { SELECTED_VOICE_NAME = null; }
  if (SELECTED_VOICE_NAME) {
    const found = voices.find(v => v.name === SELECTED_VOICE_NAME);
    if (found) { SELECTED_VOICE = found; return found; }
  }
  // prefer voices matching common male names or explicit 'Male' token
  const maleRe = /male|david|daniel|alex|mark|john|paul|henry|michael|tom|richard|mike|adam|jonathan/i;
  // prefer language close to user's navigator language
  const userLang = (navigator.language || navigator.userLanguage || 'en-US').toLowerCase();
  // 1) same language & male-looking name
  let candidate = voices.find(v => (v.lang || '').toLowerCase().startsWith(userLang.split('-')[0]) && maleRe.test(v.name));
  // 2) any male-looking name
  if (!candidate) candidate = voices.find(v => maleRe.test(v.name));
  // 3) any en-US voice
  if (!candidate) candidate = voices.find(v => (v.lang || '').toLowerCase().startsWith('en-us'));
  // 4) fallback to first available
  if (!candidate) candidate = voices[0];
  if (candidate) {
    SELECTED_VOICE = candidate;
    try { localStorage.setItem('tts_voice', candidate.name); } catch (e) {}
  }
  return candidate;
}

// Some browsers populate voices asynchronously
if ('speechSynthesis' in window) {
  // try immediate pick
  pickPreferredMaleVoice();
  // listen for updates
  window.speechSynthesis.onvoiceschanged = () => {
    pickPreferredMaleVoice();
  };
}

function normalizeKey(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

// Find an animation filename from ANIM_FILES that best matches a given name.
// Returns the filename (e.g. 'Talking.fbx') or null when not found.
function findAnimByName(name) {
  if (!name) return null;
  const want = normalizeKey(name.replace(/\.fbx$/i, ''));
  if (Array.isArray(ANIM_FILES) && ANIM_FILES.length) {
    // 1) exact normalized filename match
    for (const f of ANIM_FILES) {
      const base = normalizeKey(f.replace(/\.fbx$/i, ''));
      if (base === want) return f;
    }
    // 2) substring match (either direction)
    for (const f of ANIM_FILES) {
      const base = normalizeKey(f.replace(/\.fbx$/i, ''));
      if (base.includes(want) || want.includes(base)) return f;
    }
  }
  return null;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement('script');
    s.src = src;
    s.onload = () => setTimeout(resolve, 0);
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

function updateRendererSize(renderer, canvas) {
  // match canvas size to its parent container computed size (works when canvas is moved)
  const parent = (canvas && canvas.parentElement) ? canvas.parentElement : avatar;
  const rect = parent.getBoundingClientRect();
  // choose a square size within reasonable bounds
  const size = Math.max(64, Math.min(1024, Math.round(Math.min(rect.width, rect.height))));
  try { renderer.setSize(size, size, false); } catch (e) {}
  if (canvas) {
    // let CSS control display size while backing buffer is square
    canvas.style.width = Math.max(48, rect.width) + 'px';
    canvas.style.height = Math.max(48, rect.height) + 'px';
  }
}

// Compute a bounding box of the loaded model and position the camera to fit it
// Accepts an optional opts object: { zoom: number } where zoom>1 zooms in (closer)
avatar3d.frameModel = function frameModel(opts = {}) {
  // Respect an explicit zoom passed in opts (e.g., from the slider). Only
  // fall back to MODEL_OVERRIDES zoom when no explicit zoom was provided.
  let zoom;
  if (typeof opts.zoom === 'number' && opts.zoom > 0) {
    zoom = opts.zoom;
  } else {
    zoom = 1.0;
    try {
      const overrides = MODEL_OVERRIDES && (MODEL_OVERRIDES['model.fbx'] || MODEL_OVERRIDES['default']);
      if (overrides && typeof overrides.zoom === 'number') zoom = overrides.zoom;
    } catch (e) {}
  }
  if (!this.root || !this.camera) return;
  try {
    // Use bounding sphere for a tighter, more consistent fit across models
    const sphere = new window.THREE.Sphere();
    const box = new window.THREE.Box3().setFromObject(this.root);
    box.getBoundingSphere(sphere);
    const center = box.getCenter(new window.THREE.Vector3());
    const radius = sphere.radius || Math.max(box.getSize(new window.THREE.Vector3()).length() * 0.5, 0.5);

    // compute distance using field of view so the sphere fits vertically
    const fov = this.camera.fov * (Math.PI / 180);
    // distance required so sphere fits into camera frustum: r / sin(fov/2)
    let distance = radius / Math.sin(fov / 2);
    // small safety factor and base offset so very small models still sit in front
    distance = distance * 1.15 + 0.5;
    // apply zoom: larger zoom -> closer camera (smaller distance)
    distance = distance / zoom;

    // reduce vertical offset so the head is not pushed too low; small upward nudge
    let yOffset = Math.min(radius * 0.25, 0.35);
    try {
      const overrides = MODEL_OVERRIDES && (MODEL_OVERRIDES['model.fbx'] || MODEL_OVERRIDES['default']);
      if (overrides && typeof overrides.vOffset === 'number') yOffset = overrides.vOffset;
    } catch (e) {}
    this.camera.position.set(center.x, center.y + yOffset, center.z + distance);
    this.camera.lookAt(center);
    if (this.camera.updateProjectionMatrix) this.camera.updateProjectionMatrix();
    // diagnostic log (kept intentionally concise)
    console.log('frameModel: distance', distance.toFixed(2), 'radius', radius.toFixed(2), 'zoom', zoom);
  } catch (e) {
    console.warn('frameModel error', e);
  }
};

function showGesturePreview(canvas) {
  if (!avatar3d.ready) return;
  // simple preview cycle through several common gestures
  const previewList = ['Talking','acknowledging','happy hand gesture','head nod yes','Thoughtful Head Nod'];
  let i = 0;
  const next = () => {
    const name = previewList[i % previewList.length];
    avatar3d.play(name, { fade: 0.15 });
    i += 1;
    if (i < previewList.length) setTimeout(next, 900);
  };
  next();
}

// Create or update a small list of buttons in the preview area to trigger registered animations
function updatePreviewControls() {
  // Preview controls disabled in production UI to avoid cluttering the preview pane.
  // If you need debug buttons, temporarily re-enable this function.
  return;
}

// initialize avatar 3D in background only on larger viewports. On mobile we
// skip loading the 3D engine to save bandwidth and CPU.
try { window.avatar3d = avatar3d; } catch (e) {}
if (typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(min-width: 901px)').matches) {
  avatar3d.init().catch(e => console.warn('Avatar 3D init failed', e));
} else {
  console.log('Avatar 3D init skipped on mobile/small screen');
}

// Debugging: allow pressing 'r' to cycle orientation fixes (useful for various FBX exports)
(() => {
  const rotations = [0, -Math.PI/2, Math.PI/2, Math.PI];
  let idx = 0;
  document.addEventListener('keydown', (ev) => {
    if (ev.key.toLowerCase() !== 'r') return;
    try {
      if (!avatar3d.root) return console.log('No root to rotate');
      idx = (idx + 1) % rotations.length;
      avatar3d.root.rotation.x = rotations[idx];
      console.log('cycle-rotation: set rotation.x to', rotations[idx]);
      avatar3d.frameModel({ zoom: 1.4 });
    } catch (e) { console.warn('rotation cycle failed', e); }
  });
})();

// helper to play a gesture when the bot posts a message
function playGestureForBotText(text) {
  if (!avatar3d.ready) return;
  const key = avatar3d.gestureForText(text);
  avatar3d.play(key);
}

// ------------------ Modal handling (Preview, Settings, Error) ------------------
function openModal(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.setAttribute('aria-hidden', 'false');
  el.classList.add('is-open');
}
function closeModal(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.setAttribute('aria-hidden', 'true');
  el.classList.remove('is-open');
}

// Open the right-side docked preview (non-modal)
function openSidePreview() {
  const el = document.getElementById('preview-side');
  if (!el) return;
  el.setAttribute('aria-hidden', 'false');
  el.classList.add('is-open');
}

function closeSidePreview() {
  const el = document.getElementById('preview-side');
  if (!el) return;
  el.setAttribute('aria-hidden', 'true');
  el.classList.remove('is-open');
}

// More robust delegated click handling for modal buttons.
document.addEventListener('click', (e) => {
  // find nearest button or modal-close element so clicks on text nodes or icons still register
  const btn = e.target.closest('button, .modal-close');
  if (!btn) return;

  if (btn.id === 'preview-btn') {
    // prefer right-side docked preview rather than modal
    openSidePreview();
    // move avatar canvas into preview body
    const canvas = document.getElementById('avatar-canvas');
    const previewBody = document.getElementById('preview-side-body') || document.getElementById('preview-body');
    if (canvas && previewBody) {
      previewBody.appendChild(canvas);
      // enlarge renderer to fit preview area and frame using stored zoom
      try { updateRendererSize(avatar3d.renderer, canvas); const z = parseFloat(localStorage.getItem('preview_zoom')) || 2.2; avatar3d.frameModel({ zoom: z }); } catch (e) {}
    }
    return;
  }

  if (btn.id === 'settings-btn') {
    openModal('settings-modal');
    // load gesture-map JSON into editor
    fetch(AVATAR_MODELS_PATH + 'gesture-map.json').then(r => r.ok ? r.text() : Promise.reject()).then(txt => {
      const ed = document.getElementById('gesture-map-editor'); if (ed) ed.value = txt;
    }).catch(() => {
      const ed = document.getElementById('gesture-map-editor'); if (ed) ed.value = localStorage.getItem('gesture_map') || '';
    });
    return;
  }

    if (btn.classList && btn.classList.contains('modal-close')) {
    const target = btn.getAttribute('data-target');
    if (target === 'preview-modal' || target === 'preview-side') {
      // Do not move the 3D canvas back into the small header avatar; just
      // resize if needed and close the preview/modal so the header image
      // remains the visible static avatar.
      try {
        const canvas = document.getElementById('avatar-canvas');
        if (canvas && avatar3d && avatar3d.renderer) updateRendererSize(avatar3d.renderer, canvas);
      } catch (e) {}
      if (target === 'preview-side') closeSidePreview(); else closeModal(target);
      return;
    }
    closeModal(target);
    return;
  }
});

// Delegated input listener for the preview zoom slider so it works regardless
// of load order. This also ensures the slider picks up stored zoom when the
// side preview is opened.
document.addEventListener('input', (e) => {
  const t = e.target;
  if (!t || t.id !== 'preview-zoom') return;
  const v = parseFloat(t.value);
  // debug log to confirm input events are firing
  try { console.log('preview-zoom input event, value=', v); } catch (e) {}
  try { localStorage.setItem('preview_zoom', String(v)); } catch (e) {}
  try {
    if (window.avatar3d && avatar3d.frameModel) {
      avatar3d.frameModel({ zoom: v });
      const canvas = document.getElementById('avatar-canvas');
      if (canvas && avatar3d.renderer) updateRendererSize(avatar3d.renderer, canvas);
    }
  } catch (err) { console.warn('delegated preview-zoom handler failed', err); }
});

// Ensure slider UI is synchronized to stored zoom when opening the side preview
const origOpenSidePreview = openSidePreview;
window.openSidePreview = function() {
  origOpenSidePreview();
  try {
    const el = document.getElementById('preview-zoom');
    const stored = parseFloat(localStorage.getItem('preview_zoom')) || 2.1;
    if (el) el.value = stored;
    if (window.avatar3d && avatar3d.frameModel) avatar3d.frameModel({ zoom: stored });
    const canvas = document.getElementById('avatar-canvas');
    if (canvas && avatar3d.renderer) updateRendererSize(avatar3d.renderer, canvas);
  } catch (e) { console.warn('openSidePreview sync failed', e); }
};

// Save gesture map (editor writes to localStorage and attempts to PUT to server if endpoint present)
document.addEventListener('click', async (e) => {
  const t = e.target;
  if (!t.matches('#gesture-save')) return;
  const txt = document.getElementById('gesture-map-editor').value;
  try {
    JSON.parse(txt); // validate
    localStorage.setItem('gesture_map', txt);
    // update runtime map immediately
    try { GESTURE_MAP = JSON.parse(txt); } catch (e) {}
    // attempt to save to server path (best-effort) - server doesn't expose write route by default
    try {
      await fetch(AVATAR_MODELS_PATH + 'gesture-map.json', { method: 'PUT', body: txt });
    } catch (e) {
      // ignore
    }
    closeModal('settings-modal');
  } catch (e) {
    showError('Invalid JSON in gesture map. Please fix before saving.');
  }
});

function showError(msg) {
  const el = document.getElementById('error-body');
  if (el) el.textContent = msg;
  openModal('error-modal');
}

// end of 3D avatar player

function addMessage(role, text, sources) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  if (role === 'bot' && Array.isArray(sources) && sources.length) {
    const uniq = [];
    const seen = new Set();
    for (const s of sources) {
      const url = s.url || s.link;
      if (!url || seen.has(url)) continue;
      seen.add(url);
      uniq.push({ title: s.title || url, url });
      if (uniq.length >= 3) break;
    }
    if (uniq.length) {
      const src = document.createElement('div');
      src.className = 'sources';
      const label = document.createElement('div');
      label.textContent = 'Sources:';
      src.appendChild(label);
      for (const u of uniq) {
        const line = document.createElement('div');
        const a = document.createElement('a');
        a.href = u.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = `${u.title}`;
        line.appendChild(a);
        src.appendChild(line);
      }
      wrap.appendChild(src);
    }
  }
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  if (role === 'bot') {
    // pick a gesture/animation for this bot message
    playGestureForBotText(text);
  }
}

function addTyping() {
  const el = document.createElement('div');
  el.className = 'msg bot typing';
  el.textContent = 'George is typing…';
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function speak(text) {
  if (!speakToggle.checked) return;
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  // slightly deeper/more natural defaults; these may be overridden by the voice
  u.rate = 1.0;
  u.pitch = 0.9;
  // apply the preferred male voice if available
  try {
    if (SELECTED_VOICE) u.voice = SELECTED_VOICE;
    else {
      const v = pickPreferredMaleVoice(); if (v) u.voice = v;
    }
  } catch (e) {}
  u.onstart = () => {
    avatar.classList.add('speaking');
    try { avatar3d.play('Talking', { fade: 0.08, loop: 'repeat' }); } catch (e) {}
    try { avatar3d.stopGestureLoop(); } catch (e) {}
  };
  u.onend = () => {
    avatar.classList.remove('speaking');
    try { avatar3d.startGestureLoop(); } catch (e) {}
  };
  u.onerror = () => {
    avatar.classList.remove('speaking');
    try { avatar3d.startGestureLoop(); } catch (e) {}
  };
  window.speechSynthesis.speak(u);
}

async function sendMessage(text) {
  addMessage('user', text);
  const typingEl = addTyping();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typingEl.remove();
    if (!res.ok) {
      addMessage('bot', data.error || 'Something went wrong.');
      return;
    }
    addMessage('bot', data.answer, data.sources);
    speak(data.answer);
  } catch (e) {
    typingEl.remove();
    addMessage('bot', 'Network error. Is the server running?');
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendMessage(text);
});

// Warm greeting
addMessage('bot', "Hey, I'm George. What would you like to know about Dynavap today?");

// Probe candidate avatar image URLs (listed in data-srcs on the img) and set the first reachable one
(function probeAvatarImage() {
  try {
    const img = document.getElementById('avatar-img');
    if (!img) return;
    const list = (img.getAttribute('data-srcs') || img.getAttribute('data-src') || '').split(',').map(s=>s.trim()).filter(Boolean);
    if (!list.length) return;
    const tryNext = (i) => {
      if (i >= list.length) return;
      const url = list[i];
      const tester = new Image();
      tester.onload = () => { try { img.src = url; } catch (e) {} };
      tester.onerror = () => { tryNext(i+1); };
      tester.src = url;
    };
    tryNext(0);
  } catch (e) { /* ignore */ }
})();
