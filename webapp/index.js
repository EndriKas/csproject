// Importing the hyper text transfer
// protocol library.
var httpm=require("http");


// Importing the socket.io library that
// enables real-time bidirectional event
// based communication.
var socketio=require("socket.io");


// Importing the express web framework.
var express=require("express");


// Importing the child_process module that
// provides an interface to interact with
// the unix system.
var cp=require("child_process");



// Creating a new express application
// and setting all the necessary session
// and server configurations for real-time
// communication.
var app=express();
var http=httpm.Server(app);
var io=socketio(http);


// Setting Jade as the default
// template engine and specifying
// the directory for the static files.
app.set("view engine","pug");
app.use(express.static(__dirname + '/public'));



// Whenever a request to the route page is detected
// render the 'index.pug' file to the user.
app.get("/",function(req,res) { res.render("index.pug"); });



// The real-time server communication.It handles
// all opened sockets and connections and performs
// the necessary computations and transmissions back
// to the client.
io.on("connect",function(socket)
{
    // Alerting new user connetions.Also
    // listening at the "update vote" socket
    // for real time vote updates based on
    // user votings.
    console.log("a user has connected");


    socket.on("diagnosis",function(data)
    {
        console.log("new diagnosis request");
        console.log(data);
        var command="../neuralnet --predict --pattern-classification \
            --normalization=yes --in-file=stdin --load-dir=../thyroidologist";
        var child=cp.exec(command,function(error,stdout,stderr)
        {
            if (error) { console.log(error); return; }
            socket.emit("results",{ prediction: stdout });
        });
        child.stdin.write("1\n");
        child.stdin.write("5\n");
        child.stdin.write(data.c0+" "+data.c1+" "+data.c2+" "+data.c3+" "+data.c4+"\n");
        child.stdin.end();
    });


    // Listening to the "disconnect" socket
    // for user discnonnection and printing the
    // necessary logs messages to the terminal.
    socket.on("disconnect",function(msg)
    {
        console.log("user disconnected");
        return;
    });
    return;
});


// Setting up the port at http://localhost:3000
// to listen for requests and serve the corresponding
// handlers to deal with them.
http.listen(3000,function()
{
    console.log("Listening at port http://localhost:3000");
    return;
});
