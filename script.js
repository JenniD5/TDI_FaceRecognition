const imageUpload = document.getElementById('imageUpload');
Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models')

]).then(start)


async function start(){
    const container = document.createElement('div')
    container.position = 'relative'
    document.body.append(container)

    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    let image
    let canvas
    document.body.append('Loaded')

    imageUpload.addEventListener('change', async ()=>{
        if (image) image.remove()
        if (canvas) canvas.remove()

        image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)
        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)

        const displaySize = {with: image.with, height: image.height}

        faceapi.matchDimesions (canvas, displaySize)

        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        const realizedDetections = faceapi.resizeResults(detections, displaySize)

        const results = realizedDetections.map(d => faceMatcher.findBestMatch(d.desciptor))

        results.forEach((results,i) =>{
            const box = realizedDetections[i].detection.box
            const drawBox = new faceapi.DrawBox(box, {label: results.toString()})
            drawBox.draw(canvas)

        })
    })
}

function loadLabeledImages(){
    const labels = ['Black Widow', 'Capitan America', 'Capitana Marvel', 'Hawkeye', 'Jime Rhodes', 'Thor', 'Tony Stark']
    return Promise.all
    (
        labels.map(async label => {
            const descriptions = []
            for (let i =1; i<= 2; i++) {
                const img = await faceapi.fetchImage(`https://mawe.mx/face/images/${label}/${i}.jpg`)
                const detections = await faceapi.detectionSingleFace(img).withFaceLandmarks().withFaceDescriptors()
                descriptions.push(detections.desciptor)

            }

            return new faceapi.labeldFaceDescriptors(label,descriptions)
        })
    )
}