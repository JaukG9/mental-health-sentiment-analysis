<<<<<<< HEAD
async function classify(){
    const text   = document.getElementById("inputText").value.trim();
    const result = document.getElementById("result");
    const btn    = document.getElementById("btn");
    
    if(!text){
        result.innerHTML = '<p class="error">Please enter some text first.</p>';
        return;
    }

    btn.disabled = true;
    result.innerHTML = '<p class="spinner">Analyzing...</p>';

    try{
        const app = await GradioClient.connect("ayaangos/mental-health-sentiment-analysis");
        
        const json = await app.predict("/predict", { text: text });

        const output = json.data[0]; 
        const topLabel = output.label;
        const confidences = output.confidences;

        const topProbItem = confidences.find(item => item.label === topLabel);
        const topProb = topProbItem ? topProbItem.confidence : confidences[0].confidence;

        let html = `<p class="winner">Predicted: ${topLabel} (${(topProb * 100).toFixed(1)}%)</p>`;

        for (const item of confidences) {
            const label = item.label;
            const prob = item.confidence;
            const pct = (prob * 100).toFixed(1);
            html += `
                <div class="bar-row">
                    <span class="bar-label">${label}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${pct}%"></div>
                    </div>
                    <span class="bar-pct">${pct}%</span>
                </div>`;
        }

        result.innerHTML = html;
    }catch(err){
        result.innerHTML = `<p class="error">Error: ${err.message}</p>`;
    }finally{
        btn.disabled = false;
    }
=======
async function classify(){
    const text   = document.getElementById("inputText").value.trim();
    const result = document.getElementById("result");
    const btn    = document.getElementById("btn");
    
    if(!text){
        result.innerHTML = '<p class="error">Please enter some text first.</p>';
        return;
    }

    btn.disabled = true;
    result.innerHTML = '<p class="spinner">Analyzing...</p>';

    try{
        const app = await GradioClient.connect("ayaangos/mental-health-sentiment-analysis");
        
        const json = await app.predict("/predict", { text: text });

        const output = json.data[0]; 
        const topLabel = output.label;
        const confidences = output.confidences;

        const topProbItem = confidences.find(item => item.label === topLabel);
        const topProb = topProbItem ? topProbItem.confidence : confidences[0].confidence;

        let html = `<p class="winner">Predicted: ${topLabel} (${(topProb * 100).toFixed(1)}%)</p>`;

        for (const item of confidences) {
            const label = item.label;
            const prob = item.confidence;
            const pct = (prob * 100).toFixed(1);
            html += `
                <div class="bar-row">
                    <span class="bar-label">${label}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${pct}%"></div>
                    </div>
                    <span class="bar-pct">${pct}%</span>
                </div>`;
        }

        result.innerHTML = html;
    }catch(err){
        result.innerHTML = `<p class="error">Error: ${err.message}</p>`;
    }finally{
        btn.disabled = false;
    }
>>>>>>> 8ba3f285aa1596262b58bdd7e07cd700250eb022
}