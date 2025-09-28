import { useCallback, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import Section from './Section';
import useViewportWidth from './utils/useViewportDims';

function App() {
    const webcamRef = useRef(null);
    const [count, setCount] = useState(0);
    const processImage = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();

        fetch('http://localhost:8000/almond', {
            method: 'POST',
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            body: JSON.stringify({ image: imageSrc }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log('Success:', data);
                setCount(data.count);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    }, [webcamRef]);

    const { width, height } = useViewportWidth();

    const nutrition = [
        {
            name: 'Total Fat',
            amount: (c) => 0.6 * c,
            dailyValue: (c) => 0.7 * c,
            unit: 'g',
        },
        {
            name: 'Saturated Fat',
            amount: (c) => 0.04 * c,
            dailyValue: (c) => 0.2 * c,
            indent: true,
            unit: 'g',
        },
        {
            name: 'Trans Fat',
            amount: 0,
            dailyValue: '',
            indent: true,
            unit: 'g',
        },
        {
            name: 'Cholesterol',
            amount: 0,
            dailyValue: '',
            unit: 'g',
        },
        {
            name: 'Sodium',
            amount: 0,
            dailyValue: '',
            unit: 'mg',
        },
        {
            name: 'Total Carbohydrate',
            amount: (c) => 0.26 * c,
            dailyValue: (c) => 0.04 * c,
            unit: 'g',
        },
        {
            name: 'Dietary Fiber',
            amount: (c) => 0.13 * c,
            dailyValue: (c) => 0.47 * c,
            unit: 'g',
            indent: true,
        },
        {
            name: 'Total Sugars',
            amount: 0,
            dailyValue: '',
            indent: true,
            unit: 'g',
        },
        {
            name: 'Includes 0g Added Sugars',
            dailyValue: 0,
            indent: true,
            skip: true,
        },
        {
            name: 'Protein',
            amount: (c) => 0.26 * c,
            dailyValue: (c) => 0.57 * c,
            unit: 'g',
        },
    ];

    const vitamins = [
        { name: 'Vitamin D', amount: 0, dailyValue: 0, unit: 'mcg' },
        { name: 'Calcium', amount: (c) => 3.95 * c, dailyValue: (c) => 0.34 * c, unit: 'mg' },
        { name: 'Iron', amount: (c) => 0.04 * c, dailyValue: (c) => 0.26 * c, unit: 'mg' },
        { name: 'Potassium', amount: (c) => 12.2 * c, dailyValue: (c) => 0.26 * c, unit: 'mg' },
        { name: 'Vitamin E', amount: (c) => 0.26 * c, dailyValue: (c) => 1.73 * c, unit: 'mg' },
        { name: 'Magnesium', amount: (c) => 2.73 * c, dailyValue: (c) => 0.65 * c, unit: 'mg' },
    ];

    return (
        <>
            <div className='max-w-[100vw] max-h-[100dvh] min-h-[100dvh] flex'>
                <div className='w-[400px] flex flex-col min-h-full justify-start items-center bg-white relative z-30 text-black p-8'>
                    <div className='border-black border-2 w-full h-full p-4'>
                        <p className='text-4xl font-bold'>Almond Facts</p>

                        <hr className='border-black border-1 mt' />
                        <div className='py-1'>
                            <p>{(count / 23).toFixed(2)} servings per handful</p>
                        </div>

                        <hr className='border-black border-4' />

                        <div className='py-1 flex flex-row justify-between items-end'>
                            <div className='flex flex-col items-start'>
                                <p>{count} almonds per handful</p>
                                <p className='text-3xl font-semibold'>Calories</p>
                            </div>
                            <div className='flex flex-col justify-end items-end min-h-full'>
                                <p className='text-5xl font-semibold'>{7 * count}</p>
                            </div>
                        </div>

                        <hr className='border-black border-3 my-1' />

                        <div className='flex flex-row justify-end items-center grow'>
                            <p className='text-sm'>% Daily Value*</p>
                        </div>

                        <hr className='border-black border-[0.5] w-full' />

                        <Section template={nutrition} data={count} />

                        <hr className='border-black border-4 my-1' />
                        <Section template={vitamins} data={count} />
                        <hr className='border-black border-4 my-1' />
                        <p className='text-sm'>
                            The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a
                            daily diet. 2,000 calories a day is used for general nutrition advice.
                        </p>
                    </div>
                </div>
                <Webcam
                    ref={webcamRef}
                    style={{ width: width - 400, height: 'auto', zIndex: 1 }}
                    onClick={processImage}
                />
            </div>
        </>
    );
}

export default App;
