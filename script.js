let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let startBtn = document.getElementById("startBtn");
let status = document.getElementById("status");
let log = document.getElementById("log");

let previousGray = null;

const BOARD_SIZE = 800;
const FRAME_DELAY = 120;

let dictionary;
let parameters;

function logMsg(msg){
console.log(msg);
log.innerText = msg + "\n" + log.innerText;
}

function waitForOpenCV(){

if(!window.cvReady){
setTimeout(waitForOpenCV,200);
return;
}

logMsg("OpenCV confirmed ready");

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50);
parameters = new cv.aruco.DetectorParameters();

}

waitForOpenCV();


startBtn.onclick = async function(){

if(!window.cvReady){
alert("OpenCV still loading");
return;
}

logMsg("Requesting camera");

const stream = await navigator.mediaDevices.getUserMedia({
video:{
facingMode:"environment",
width:{ideal:640},
height:{ideal:640}
}
});

video.srcObject = stream;

video.onloadedmetadata = ()=>{

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

status.innerText="Camera running";

logMsg("Camera started");

setTimeout(processFrame,FRAME_DELAY);

};

};


function detectBoard(src){

let markerCorners = new cv.MatVector();
let markerIds = new cv.Mat();

cv.aruco.detectMarkers(src,dictionary,markerCorners,markerIds,parameters);

logMsg("Markers detected "+markerIds.rows);

if(markerIds.rows<4){
markerCorners.delete();
markerIds.delete();
return null;
}

let boardCorners={};

for(let i=0;i<markerIds.rows;i++){

let id = markerIds.intAt(i,0);
let corner = markerCorners.get(i);

let x = corner.data32F[0];
let y = corner.data32F[1];

boardCorners[id]=[x,y];

}

markerCorners.delete();
markerIds.delete();

if(boardCorners[0] && boardCorners[1] && boardCorners[2] && boardCorners[3]){

let srcTri = cv.matFromArray(4,1,cv.CV_32FC2,[

boardCorners[0][0],boardCorners[0][1],
boardCorners[1][0],boardCorners[1][1],
boardCorners[2][0],boardCorners[2][1],
boardCorners[3][0],boardCorners[3][1]

]);

let dstTri = cv.matFromArray(4,1,cv.CV_32FC2,[

0,0,
BOARD_SIZE,0,
BOARD_SIZE,BOARD_SIZE,
0,BOARD_SIZE

]);

let M = cv.getPerspectiveTransform(srcTri,dstTri);

let warped = new cv.Mat();

cv.warpPerspective(src,warped,M,new cv.Size(BOARD_SIZE,BOARD_SIZE));

srcTri.delete();
dstTri.delete();
M.delete();

logMsg("Board warp success");

return warped;

}

return null;
}


function processFrame(){

ctx.drawImage(video,0,0,canvas.width,canvas.height);

let src = cv.imread(canvas);

let gray = new cv.Mat();
cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);

let board = detectBoard(src);

if(board){

let display = new cv.Mat();

cv.resize(board,display,new cv.Size(canvas.width,canvas.height));

cv.imshow(canvas,display);

display.delete();
board.delete();

status.innerText="Board detected";

}

src.delete();
gray.delete();

setTimeout(processFrame,FRAME_DELAY);

}
