import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from loopnet import compile_structured


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node is needed for activation capture checks')
class ActivationCaptureTests(unittest.TestCase):
    def node(self, script, data=None):
        result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', script],
                                cwd=ROOT, input=json.dumps(data), text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_preserves_input_and_output_across_multiple_recurrences(self):
        net = compile_structured('total=0\nwhile counter > 0:\n    total+=counter\n    counter-=1\nresult=relu(total-1)')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'loop.json'
            net.save(path)
            self.node("""
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {captureActivations} from './loopnet/web/activation-trace.mjs';
const data=JSON.parse(fs.readFileSync(0,'utf8'));
const n=new ResidualNetwork(JSON.parse(fs.readFileSync(data.model,'utf8')));
const input=n.initialState({counter:3}), original=input.slice(), expected=input.slice();
const calls=n.run(expected), trace=captureActivations(n,input);
assert.ok(calls>1);
assert.equal(trace.calls,calls);
assert.equal(trace.stages.length,1+data.depth*calls);
assert.deepEqual(input,original);
assert.deepEqual(trace.residuals.at(-1),expected);
assert.equal(n.read(expected,'result'),5);
assert.equal(trace.stages.at(-1).recurrence,calls);
assert.equal(trace.stages[0].phase,'input');
assert.equal(trace.channels[n.model.terminate_channel],'terminate');
for(const stage of trace.stages) {
 assert.ok(stage.values instanceof Float64Array);
 assert.ok(stage.residual<trace.residuals.length);
 if(stage.phase==='hidden')assert.equal(stage.values.length,n.blocks[stage.block].layers[stage.layer].count);
}
// Capture leaves reusable hidden buffers safe for the next ordinary run.
const repeated=original.slice();n.run(repeated);assert.deepEqual(repeated,expected);
""", {'model': str(path), 'depth': net.summary()['feedforward_layers']})

    def test_images_contain_signed_reglu_outputs_before_output_projection(self):
        self.node("""
import assert from 'node:assert/strict';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {captureActivations} from './loopnet/web/activation-trace.mjs';
const row=bias=>({bias,terms:[]});
const n=new ResidualNetwork({format:'loopnet.reglu.v4',variables:['x'],width:8,schedule:[0],
 terminate_channel:1,stack_channel:2,jump_channel:3,index_channel:4,margin_channel:5,
 blocks:[{bias:[],units:[
  {gate:row(2),value:row(-3),outputs:[[0,7]]},
  {gate:row(-1),value:row(99),outputs:[[0,100]]},
  {gate:row(1),value:row(1),outputs:[[1,1]]}]}]});
const trace=captureActivations(n,n.initialState({x:3}));
assert.equal(trace.stages.length,2);
assert.deepEqual(Array.from(trace.stages[1].values),[-6,0,1]);
assert.equal(trace.residuals.at(-1)[0],-39); // 3 + 7*(-6), distinct from the hidden activation
assert.equal(trace.maximum,39);
""")

    def test_heatmap_scale_preserves_sign_and_is_shared_between_stages(self):
        self.node("""
import assert from 'node:assert/strict';
import {activationImage} from './loopnet/web/activation-trace.mjs';
const a=activationImage(Float64Array.from([0,1,-1,100]),100);
const b=activationImage(Float64Array.from([1]),100);
assert.deepEqual(Array.from(a.data.slice(0,4)),[11,16,15,255]);
assert.deepEqual(a.data.slice(4,8),b.data.slice(0,4));
assert.ok(a.data[5]>a.data[4]); // positive is green
assert.ok(a.data[8]>a.data[9]); // negative is orange
assert.ok(a.data[13]>a.data[5]); // larger magnitude is brighter
assert.ok(a.width*a.height>=4);
assert.equal(activationImage(new Float64Array(),0).data.length,4);
""")

    def test_residual_detail_separates_pixels_and_ignores_control_addresses_for_color_range(self):
        self.node("""
import assert from 'node:assert/strict';
import {residualOverview,activationImage} from './loopnet/web/activation-trace.mjs';
const trace={channels:['frame[0]','frame[1]','player_x','ammo','pc','terminate'],variableCount:4,
 pixels:Int32Array.from([0,1]),
 residuals:[Float64Array.from([0,0,4,60,0,0]),Float64Array.from([8,10,4,59,2000000,1])],
 stages:[{phase:'input'},{phase:'hidden'},{phase:'clear'},{phase:'write'}]};
const view=residualOverview(trace);
assert.deepEqual(Array.from(view.indices),[2,3,4,5]);
assert.equal(view.maximum,60);
assert.equal(view.writeback,3);
assert.equal(trace.residuals[1][4],2000000); // only colors saturate; values remain exact
const state=activationImage(Float64Array.from([1,2000000]),view.maximum);
const old=activationImage(Float64Array.from([1]),2000000);
assert.ok(state.data[1]>old.data[1]);
assert.deepEqual(Array.from(state.data.slice(4,8)),[195,237,140,255]);
""")

    def test_residual_detail_handles_empty_programs_and_legacy_residual_layers(self):
        self.node("""
import assert from 'node:assert/strict';
import {residualOverview,activationStatistics} from './loopnet/web/activation-trace.mjs';
const trace={channels:['terminate'],variableCount:0,residuals:[Float64Array.from([0]),Float64Array.from([1])],
 stages:[{phase:'input'},{phase:'residual'}]};
assert.deepEqual(Array.from(residualOverview(trace).indices),[0]);
assert.equal(residualOverview(trace).maximum,1);
assert.equal(residualOverview(trace).writeback,1);
assert.deepEqual(activationStatistics(new Float64Array()),{minimum:0,maximum:0,active:0,count:0});
""")

    def test_live_atlas_records_actual_outputs_and_reuses_cleared_buffers(self):
        self.node("""
import assert from 'node:assert/strict';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {captureActivations} from './loopnet/web/activation-trace.mjs';
import {ActivationAtlas,layoutActivationColumns} from './loopnet/web/activation-atlas.mjs';
const row=(bias,terms=[])=>({bias,terms});
const n=new ResidualNetwork({format:'loopnet.reglu.v4',variables:['x'],width:8,schedule:[0],
 terminate_channel:1,stack_channel:2,jump_channel:3,index_channel:4,margin_channel:5,
 blocks:[{bias:[],units:[
  {gate:row(0,[[0,1]]),value:row(-3),outputs:[[0,1]]},
  {gate:row(1),value:row(1),outputs:[[1,1]]}]}]});
const atlas=new ActivationAtlas(n), buffers=[...atlas.buffers.values()];
for(const x of [2,-1]) {
 const state=n.initialState({x}), trace=captureActivations(n,state);
 atlas.begin(state);n.run(state,5000,atlas);
 assert.deepEqual(state,trace.residuals.at(-1));
 for(const tile of atlas.layout.tiles)assert.deepEqual(
  atlas.values.slice(tile.offset,tile.offset+tile.count),trace.stages[tile.stage].values);
 const image=atlas.render();
 const seen=new Set();
 for(let i=0;i<atlas.values.length;i++) {
  const at=atlas.locations[i];assert.ok(at<image.width*image.height);assert.ok(!seen.has(at));seen.add(at);
  const value=atlas.values[i],pixel=Array.from(image.data.slice(at*4,at*4+4));
  assert.equal(pixel[3],255);
  if(value===0)assert.deepEqual(pixel,[0,0,0,255]);
  else if(value>0)assert.ok(pixel[1]>pixel[0]&&pixel[1]===pixel[2]); // cyan
  else assert.ok(pixel[0]>pixel[1]&&pixel[1]===pixel[2]); // red
 }
 assert.deepEqual(Array.from(image.data.slice(0,4)),[0,0,0,255]); // gaps are black too
 assert.equal(seen.size,atlas.values.length);
 const transfer=structuredClone(image,{transfer:[image.data.buffer]});
 assert.equal(image.data.byteLength,0);
 const returned=structuredClone(transfer.data.buffer,{transfer:[transfer.data.buffer]});
 assert.equal(atlas.render(returned).data.buffer,returned);
 assert.equal(atlas.inspect(1,0),x>0?-6:0); // inactive gates clear the previous frame
}
assert.deepEqual([...atlas.buffers.values()],buffers);
for(let i=0;i<buffers.length;i++)assert.equal([...atlas.buffers.values()][i],buffers[i]);
const packed=layoutActivationColumns([0,1,4,1000,62430].map(count=>({count})));
for(const a of packed.tiles) {
 assert.ok(a.x>=0&&a.y>=0&&a.x+a.width<=packed.width&&a.y+a.height<=packed.height);
 for(const b of packed.tiles)if(a!==b)assert.ok(
  a.x+a.width<=b.x||b.x+b.width<=a.x||a.y+a.height<=b.y||b.y+b.height<=a.y);
}
assert.deepEqual(layoutActivationColumns([]),{width:1,height:1,count:0,visibleCount:0,tiles:[]});
""")

    def test_atlas_columns_truncate_display_without_truncating_computation(self):
        self.node("""
import assert from 'node:assert/strict';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {ActivationAtlas,layoutActivationColumns,atlasNeuronAt} from './loopnet/web/activation-atlas.mjs';
const counts=[61,2,60,121,0],layout=layoutActivationColumns(counts.map(count=>({count})));
assert.deepEqual(layout.tiles.map(t=>t.count),counts); // execution order, never size sorting
assert.deepEqual(layout.tiles.map(t=>t.width),[1,1,1,1,1]);
assert.deepEqual(layout.tiles.map(t=>t.visibleCount),[60,2,60,60,0]);
assert.equal(layout.visibleCount,182);
for(const tile of layout.tiles) {
 assert.ok(tile.height<=60);
 for(let neuron=0;neuron<tile.visibleCount;neuron++) {
  const hit=atlasNeuronAt(layout,tile.x,tile.y+neuron);
  assert.equal(hit.tile.stage,tile.stage);assert.equal(hit.neuron,neuron);
 }
}
const first=layout.tiles[0];
assert.equal(atlasNeuronAt(layout,first.x,first.y+59).neuron,59);
assert.equal(atlasNeuronAt(layout,first.x,first.y+60),null);
assert.equal(atlasNeuronAt(layout,first.x+1,first.y),null); // no overflow column
assert.equal(atlasNeuronAt(layout,0,0),null);
const expanded=layoutActivationColumns([{count:201}],{neuronLimit:200});
assert.equal(expanded.tiles[0].visibleCount,200);
assert.equal(atlasNeuronAt(expanded,1,200).neuron,199);
assert.equal(atlasNeuronAt(expanded,1,201),null);
const sampled=layoutActivationColumns(counts.map(count=>({count})),{neuronLimit:100,layerStride:3});
assert.deepEqual(sampled.tiles.filter(t=>t.width).map(t=>t.stage),[0,3]);
assert.deepEqual(sampled.tiles.map(t=>t.offset),[0,61,63,123,244]); // skipped layers still occupy recorded state
assert.equal(sampled.count,244);assert.equal(sampled.width,5);
assert.equal(sampled.tiles[3].visibleCount,100);
assert.equal(atlasNeuronAt(sampled,3,100).tile.stage,3);
assert.equal(atlasNeuronAt(sampled,3,100).neuron,99);
// The omitted 61st neuron still computes and contributes to the neural result.
const row=bias=>({bias,terms:[]});
const units=Array.from({length:61},()=>({gate:row(1),value:row(1),outputs:[]}));
units[59].outputs=[[1,1]];
units[60]={gate:row(1),value:row(1e9),outputs:[[0,2]]};
const n=new ResidualNetwork({format:'loopnet.reglu.v4',variables:['x'],width:8,schedule:[0],
 terminate_channel:1,stack_channel:2,jump_channel:3,index_channel:4,margin_channel:5,
 blocks:[{bias:[],units}]});
const a=new ActivationAtlas(n),state=n.initialState({x:7});a.begin(state);n.run(state,5000,a);
assert.equal(n.read(state,'x'),2000000007);
assert.equal(a.values.at(-1),1e9);
assert.equal(a.inspect(1,59),1);
assert.equal(a.inspect(1,60),null);
const image=a.render();
assert.equal(image.count,68);
assert.equal(image.maximum,7); // unseen tail does not dim the visible neurons
assert.equal(a.visibleIndices.length,68);
for(let index=0;index<a.visibleIndices.length;index++) {
 const position=a.locations[index],hit=atlasNeuronAt(a.layout,position%a.layout.width,Math.floor(position/a.layout.width));
 assert.equal(hit.tile.offset+hit.neuron,a.visibleIndices[index]);
}
// Changing the display cap reveals actual recorded activations without rerunning
// or replacing any neural state or recording buffers, then restores the old range.
const values=a.values,buffers=[...a.buffers.values()],final=state.slice();
a.setNeuronLimit(200);
assert.equal(a.inspect(1,60),1e9);
assert.equal(a.render().maximum,1e9);
assert.equal(a.values,values);
for(let i=0;i<buffers.length;i++)assert.equal([...a.buffers.values()][i],buffers[i]);
assert.deepEqual(state,final);
a.setNeuronLimit(60);
assert.equal(a.inspect(1,60),null);
assert.equal(a.render().maximum,7);
a.setNeuronLimit(100,3);
assert.equal(a.visibleIndices.length,8); // this fixture has only an input and one hidden stage
assert.equal(a.values,values);assert.equal(a.values.at(-1),1e9);
assert.equal(a.inspect(1,59),null); // the skipped layer has no display cells
a.setNeuronLimit(60);
assert.equal(a.inspect(1,59),1); // restoring the normal view keeps the same activations
assert.throws(()=>a.setNeuronLimit(0),/positive integer/);
""")

    def test_atlas_normalizes_each_current_layer_without_changing_activations(self):
        self.node("""
import assert from 'node:assert/strict';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {ActivationAtlas} from './loopnet/web/activation-atlas.mjs';
const unit=scale=>({gate:{bias:1,terms:[]},value:{bias:0,terms:[[0,scale]]},outputs:[]});
const n=new ResidualNetwork({format:'loopnet.reglu.v4',variables:['x'],width:8,schedule:[0,1,2],
 blocks:[{bias:[],units:[unit(1e9),unit(-1e9),unit(0)]},
         {bias:[],units:[unit(1e-6),unit(-1e-6),unit(0)]},
         {bias:[],units:[unit(0)]}]});
const atlas=new ActivationAtlas(n);
for(const x of [1e-3,1e-6,0,Number.MIN_VALUE]) {
 const state=n.initialState({x}),before=state.slice();
 atlas.begin(state);n.forward(state,atlas);
 const values=atlas.values.slice(),image=atlas.render();
 const pixel=(stage,neuron)=>{
  const tile=atlas.layout.tiles[stage],at=((tile.y+neuron)*image.width+tile.x)*4;
  return Array.from(image.data.slice(at,at+4));
 };
 assert.deepEqual(state,before);assert.deepEqual(atlas.values,values);
 for(const stage of [1,2]) {
  // Tiny and huge layers both reach full brightness, even after a darker frame.
  const nonzero=atlas.values[atlas.layout.tiles[stage].offset]!==0;
  assert.deepEqual(pixel(stage,0),nonzero?[80,220,220,255]:[0,0,0,255]);
  assert.deepEqual(pixel(stage,1),nonzero?[240,70,70,255]:[0,0,0,255]);
  assert.deepEqual(pixel(stage,2),[0,0,0,255]);
 }
 assert.deepEqual(pixel(3,0),[0,0,0,255]);
 if(x===0)assert.equal(image.maximum,0);
}
""")

    def test_live_atlas_matches_every_transient_stage_across_recurrences(self):
        net = compile_structured('total=0\nwhile counter > 0:\n    total+=counter\n    counter-=1\nresult=relu(total-1)')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'atlas.json'
            net.save(path)
            self.node("""
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {captureActivations} from './loopnet/web/activation-trace.mjs';
import {ActivationAtlas} from './loopnet/web/activation-atlas.mjs';
const data=JSON.parse(fs.readFileSync(0,'utf8'));
const n=new ResidualNetwork(JSON.parse(fs.readFileSync(data.model,'utf8'))),atlas=new ActivationAtlas(n);
for(const counter of [3,0]) {
 const state=n.initialState({counter}),trace=captureActivations(n,state);
 atlas.begin(state);const calls=n.run(state,5000,atlas);
 assert.deepEqual(state,trace.residuals.at(-1));
 assert.equal(atlas.recurrence,calls);
 assert.equal(atlas.layout.tiles.length,1+(trace.stages.length-1)/calls);
 for(const tile of atlas.layout.tiles) {
  const stage=tile.stage===0?trace.stages[0]:trace.stages.at(tile.stage-atlas.layout.tiles.length);
  assert.deepEqual(atlas.values.slice(tile.offset,tile.offset+tile.count),stage.values);
 }
}
""", {'model': str(path)})


if __name__ == '__main__':
    unittest.main()
