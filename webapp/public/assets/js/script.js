$(function()
{
    var socket=io.connect();
    $("#diagnose").on("click",function()
    {
        $("#result").html("");
        $("#loader").show(); var list=[];
        $("input[type='text']").each(function() { list.push(this.value); });
        var data={ c0: list[0], c1: list[1], c2: list[2], c3:list[3], c4: list[4] };
        socket.emit("diagnosis",data);
    });

    socket.on("results",function(data)
    {
        setTimeout(function ()
        {
            var class1="euthyroidism"; var class2="hyperthyroidism";
            var class3="hypothyroidism"; data.prediction=$.trim(data.prediction);
            if (data.prediction.localeCompare("1 0 0")==0) { $("#result").html(class1); }
            if (data.prediction.localeCompare("0 1 0")==0) { $("#result").html(class2); }
            if (data.prediction.localeCompare("0 0 1")==0) { $("#result").html(class3); }
            $("#loader").hide();
        },1000);
    });
});
