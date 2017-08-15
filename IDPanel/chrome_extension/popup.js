function toTitleCase(str)
{
    return str.replace(/\w\S*/g, function(txt){return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();});
}

function format_panel_line(uri, scores){
    var scores_to_display = [];
    for(var label in scores){
        if (scores.hasOwnProperty(label)){
            if (scores[label] > -1){
                scores_to_display.push([label, scores[label]]);
            }
        }
    }
    scores_to_display.sort(function (a, b){
       return b[1] - a[1];
    });


    var div = document.createElement("div");
    div.setAttribute("class", "panel_item");
    div.textContent = uri;

    scores_to_display.forEach(function(item, index, list){
        var d = document.createElement("div");
        d.setAttribute("class", "panel_subitem");
        d.textContent = toTitleCase(item[0]).concat(": ").concat((item[1]).toString());
        div.appendChild(d);
    });

    return div;
}

function updateBots() {
    chrome.storage.sync.get("panels", function(objects){
        console.log("Fetched bot panels from local storage");
        if ("panels" in objects) {
            var sandbox = document.getElementById('panel_list');
            while (sandbox.firstChild){
                sandbox.removeChild(sandbox.firstChild);
            }
            for(var uri in objects["panels"]) {
                if (objects["panels"].hasOwnProperty(uri)) {
                    sandbox.appendChild(format_panel_line(uri, objects["panels"][uri]));
                }
            }
            //sandbox.appendChild(list);
        } else {
            var sandbox = document.getElementById('panel_list');
            while (sandbox.firstChild){
                sandbox.removeChild(sandbox.firstChild);
            }
            sandbox.textContent = "No panels identified";
        }
    });
}

chrome.storage.onChanged.addListener(function(changes, namespace) {
    updateBots();
});


document.addEventListener('DOMContentLoaded', function() {
    updateBots();

    document.getElementById('clear-button').addEventListener('click', function() {
        chrome.storage.sync.clear(function(objects) {
            updateBots();
            var port = chrome.extension.connect({name: "..."});
            port.postMessage("clear...");
        });
    });

    document.getElementById('update-button').addEventListener('click', function() {
        updateBots();
    });
});