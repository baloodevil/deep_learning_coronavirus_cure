<template>
  <div id="app">


    <div v-for="(weight, index) in model.weights" :key="index">
      {{weight}}

      <br />
    </div>

  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  name: 'App',
  data() {
    return {
      msg: "Hello World!",
      model: "",
    }
  },
  created(){
    this.train();
  },
  methods: {
    train: function() {
      this.model = tf.sequential();
      this.model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
      this.model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

      const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
      const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

      this.model.fit(xs, ys, { epochs: 500 });

      this.msg = this.model.predict(tf.tensor2d([10], [1, 1]));

      console.log(this.model);

      // var woo = this.model.getWeights();
      // console.log(woo);

      var poo = this.model.layers[0].getWeights();
      poo[0].print();
      console.log(poo[0]);

    },

    // getLuminance: function(hex, lum) {
    //   // validate hex string
    //   hex = String(hex).replace(/[^0-9a-f]/gi, '');
    //   if (hex.length < 6) {
    //     hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    //   }
    //   lum = lum || 0;

    //   // convert to decimal and change luminosity
    //   var rgb = "#", c, i;
    //   for (i = 0; i < 3; i++) {
    //     c = parseInt(hex.substr(i*2,2), 16);
    //     c = Math.round(Math.min(Math.max(0, c + (c * lum)), 255)).toString(16);
    //     rgb += ("00"+c).substr(c.length);
    //   }

    //   return rgb;
    // },
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 10px;
}
.box {
  width: 100px;
  height: 100px;
  background-color: rgb(196, 196, 226);
  border-color: rgb(82, 82, 243);
  border-width: 2px;
  border-style: solid;
  border-radius: 5px;
}
</style>