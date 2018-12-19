var canvas_num = 0;
var flag = false;
var array = [];


function init() {
    var canvas = document.getElementById("the_canvas");
    var ctx = canvas.getContext("2d");
    canvas.width = 240;
    canvas.height = 240;
}
window.onload = init();

document.getElementById('generate-button').onclick = ui_generate_button_event_listener;

// ノイズから画像を生成する関数
function ui_generate_button_event_listener(event) {
    flag = true;
    tf.loadModel('/web/model/model.json').then(handleModel).catch(handleError);
    function handleModel(model) {
        const y = tf.tidy(() => {
            const z = tf.randomNormal([1, 100]);
            var y = model.predict(z).squeeze().div(tf.scalar(2)).add(tf.scalar(0.5));
            return y;
        });
        array.push(y)

        let c = document.getElementById("the_canvas");
        tf.toPixels(image_enlarge(y, 4), c);
        history();
    }
    function handleError(error) {
        console.log("model error")
    }

}

// 画像を拡大する関数
function image_enlarge(y, draw_multiplier) {
    if (draw_multiplier === 1) {
        return y;
    }
    let size = y.shape[0];
    return y.expandDims(2).tile([1, 1, draw_multiplier, 1]
    ).reshape([size, size * draw_multiplier, 3]
    ).expandDims(1).tile([1, draw_multiplier, 1, 1]
    ).reshape([size * draw_multiplier, size * draw_multiplier, 3])
}

// 画像の生成履歴の表示
function history() {
    if (flag == false) return;
    var canvas = document.createElement('canvas')
    canvas.id = String(canvas_num)
    document.getElementById('history').appendChild(canvas);
    c = document.getElementById(String(canvas_num));
    tf.toPixels(image_enlarge(array[canvas_num], 2), c);
    canvas_num += 1;
}
