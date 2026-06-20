const FB_URL = 'https://mental-health-status-analysis-default-rtdb.firebaseio.com/classifyCount.json'

async function incrementCount(){
    const res = await fetch(FB_URL);
    const current = (await res.json()) || 0;
    const updated = current + 1;
    await fetch(FB_URL, {
        method: 'PUT',
        body: JSON.stringify(updated)
    });
    document.getElementById('use-count').textContent = `This tool has been used ${updated.toLocaleString()} times.`;
}

async function loadCount(){
    const res = await fetch(FB_URL);
    const count = (await res.json()) || 0;
    document.getElementById('use-count').textContent = `This tool has been used ${count.toLocaleString()} times.`;
    console.log(count);
}
loadCount()

async function classify(){
    const text = document.getElementById("inputText").value.trim();
    const result = document.getElementById("result");
    const btn = document.getElementById("btn");
    
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

        await incrementCount();
    }catch(err){
        result.innerHTML = `<p class="error">Error: ${err.message}</p>`;
    }finally{
        btn.disabled = false;
    }
    
    const textLength = textarea.value.trim().length;
    if(textLength > 0 && textLength < 100){
        warning.classList.remove("hidden");
    }
}

const textarea = document.getElementById("inputText");
const charCount = document.getElementById("char-count");
const warning = document.getElementById("length-warning");

textarea.addEventListener('input', () => {
    const textLength = textarea.value.trim().length;
    charCount.textContent = textLength;
    warning.classList.add("hidden");
});