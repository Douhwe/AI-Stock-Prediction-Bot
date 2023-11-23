import './App.css';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart, CategoryScale, LineController, LineElement, PointElement, LinearScale } from 'chart.js';
import { Tooltip } from 'chart.js';
import 'chartjs-adapter-moment';
import { TimeScale } from 'chart.js';


Chart.register(CategoryScale, LineController, LineElement, PointElement, LinearScale, TimeScale, Tooltip);

function App() {
    const [selectedStock, setSelectedStock] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [historicalData, setHistoricalData] = useState([]);

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);


    function getTomorrowFormattedDate() {
        const date = new Date();
        date.setDate(date.getDate() + 1); 
    
        const month = date.getMonth() + 1; 
        const day = date.getDate();
        const year = date.getFullYear();
    
        return `${month}/${day}/${year}`;
    }
    
    useEffect(() => {
        if (selectedStock) {
            const fetchHistoricalData = async () => {
                try {
                    const response = await axios.get(`http://localhost:5000/stocks/${selectedStock}`);
                    console.log("Historical Data Response:", response.data);
    
                    if (response.data && response.data.originalData) {
                        setHistoricalData(response.data.originalData);
                    } else {
                        console.error("Original data not found in response");
                        setHistoricalData([]);
                    }
                } catch (error) {
                    console.error("Error fetching historical data:", error);
                    setHistoricalData([]);
                }
            };
    
            fetchHistoricalData();
        }
    }, [selectedStock]);

    const fetchPrediction = async () => {
        setIsLoading(true); 
        try {
            const response = await axios.get(`http://localhost:5000/predict/${selectedStock}`);
            console.log("Prediction Data Response:", response.data); // Log prediction data
            if (response.data && response.data.predicted_closing_price) {  // changed 'prediction' to 'predicted_closing_price'
                setPrediction(response.data.predicted_closing_price);  
            }
        } catch (error) {
            console.error("Error fetching the prediction:", error);
            setError('Failed to fetch prediction.');
        }
        setIsLoading(false);
    };

    return (
        <div className="App">
            <h1>AI Trading Prediction Bot</h1>
    
            <select value={selectedStock} onChange={e => {
                setSelectedStock(e.target.value)}}>
                
                <option value="">Select a stock</option>
                <option value="AAPL">Apple Inc. - AAPL</option>
                <option value="GOOGL">Alphabet Inc. - GOOGL</option>
                <option value="MSFT">Microsoft Corporation - MSFT</option>
                <option value="AMZN">Amazon Inc. - AMZN</option>
                <option value="TSLA">Tesla Inc. - TSLA</option>
                <option value="NVDA">NVIDIA Inc. - NVDA</option>
                <option value="META">META Inc. - META</option>
                <option value="WMT">Walmart Inc. - WMT</option>
                <option value="JPM">JPMorgan Chase - JPM</option>
            </select>
    
            <button onClick={fetchPrediction}>Get Prediction</button>
    
            {isLoading && <p>Loading...</p>}  {/* Display loading message when data is being fetched */}
            {error && <p style={{ color: 'red' }}>{error}</p>}  {/* Display any errors in red for visibility */}
    
            {prediction && (
                <div>
                    <h2>Prediction for {selectedStock} on {getTomorrowFormattedDate()}</h2>
                    <p>{prediction}</p>
                </div>
            )}
    
    {historicalData.length > 0 && (
    <div>
        <h2>Data for {selectedStock}</h2>

        <div className="chart-container">
        <Line 
            data={{
                labels: historicalData.map(data => data.Date), // Make sure 'date' is the correct property
                datasets: [
                    {
                        label: 'Closing Price',
                        data: historicalData.map(data => data.Close), // Make sure 'closing_price' is the correct property
                        borderColor: 'blue',
                        fill: false
                    },
                    prediction && {
                        label: 'Prediction',
                        data: [...historicalData.map(data => data.Close), prediction],
                        borderColor: 'red',
                        borderDash: [5, 5],
                        fill: false
                    }
                ].filter(Boolean)
            }}
            options={{
                scales: {
                    x: {
                        type: 'time', 
                        time: {
                            unit: 'month', 
                            tooltipFormat: 'll', // Format of the tooltip
                        },
                        ticks: {
                            maxRotation: 0, // Prevents the labels from rotating
                            autoSkip: true, // Allows Chart.js to skip labels to avoid overlapping
                            maxTicksLimit: 20 // Limits the maximum number of ticks displayed
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        type: 'linear',
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Price of Single Share (USD)'
                        }
                    },
                },

                elements: {
                    point: {
                        radius: 3, // Default point radius
                        hoverRadius: 10, // Radius of point on hover
                    }
                },

                plugins: {
                    tooltip: {
                        enabled: true,
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                }
                                return label;
                            },
                            afterLabel: function(context) {
                                const date = context.label;
                                return `Date: ${date}`;
                            }
                        }
                    }
                },
                maintainAspectRatio: true, // Add this to control the aspect ratio
                responsive: true // Ensure the chart is responsive to window size
            }}
        />
        </div>
    </div>
)}

        </div>
    );
    }
export default App;
