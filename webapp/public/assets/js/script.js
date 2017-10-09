/**
 * This minimalistic script,controls the
 * receiving and sending of data packges
 * from the client to the server and vice
 * versa.It uses sockets for real-time
 * communication and jquery for simple
 * dom manipulation.
 */
$(document).ready(function()
{
    // Initializing the socket io procedure.
    var socket=io.connect();

    // Attaching an event listener to the
    // button that has id 'diagnosis'.
    $("#diagnose").on("click",function()
    {
        $("#result").html(""); // clean up any previously displayed results.
        var list=[]; // start the gif loader.
        // read the user provided values from the input attributes.
        $("input[type='text']").each(function() { list.push(this.value); });
        // format the read values as json.

        for (var i=0;i<list.length;i++)
        {
            if (!$.isNumeric(list[i])) { alert("Data must be numerical"); return; }
        }

        var data={ c0: list[0], c1: list[1], c2: list[2], c3:list[3], c4: list[4] };
        socket.emit("diagnosis",data);  // Send the input values to the server via sockets.
    });

    // Listen on the results socket for
    // output values from the server.
    socket.on("results",function(data)
    {
        // Format the received output result and based on it's value
        // write the corresponding classifcation into the label that
        // has the id 'result'.
        var class1="euthyroidism"; var class2="hyperthyroidism";
        var class3="hypothyroidism"; data.prediction=$.trim(data.prediction);

        if (data.prediction.localeCompare("1 0 0")==0)
        {
            $("#result").html(class1);
            $("#result").removeClass('text-success');
            $("#result").removeClass('text-danger');
            $("#result").addClass('text-success');
            $("#result").attr('href', 'https://en.wikipedia.org/wiki/Euthyroid');
        }

        if (data.prediction.localeCompare("0 1 0")==0)
        {
            $("#result").html(class2);
            $("#result").removeClass('text-success');
            $("#result").removeClass('text-danger');
            $("#result").addClass('text-danger');
            $("#result").attr('href', 'https://en.wikipedia.org/wiki/Hyperthyroidism');
        }

        if (data.prediction.localeCompare("0 0 1")==0)
        {
            $("#result").html(class3);
            $("#result").removeClass('text-success');
            $("#result").removeClass('text-danger');
            $("#result").addClass('text-danger');
            $("#result").attr('href', 'https://en.wikipedia.org/wiki/Hypothyroidism');
        }

        $("#result").attr("target","_blank");
    });
});
