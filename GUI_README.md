# Network IDS GUI - SCS-ID vs Baseline CNN Testing Interface

This GUI application provides an interactive testing environment for comparing the performance of the SCS-ID model against the Baseline CNN for network intrusion detection.

## 🚀 Quick Start

### Launch the GUI:
```bash
python launch_gui.py
```

Or directly:
```bash
python network_ids_gui.py
```

## 📱 Features

### 1. **Single Sample Testing**
- Generate individual network flow samples
- Select attack type and anomaly level
- Test both models simultaneously
- View detailed predictions and confidence scores
- Compare model performance on specific samples

### 2. **Batch Testing** 
- Generate batches of network samples (configurable size)
- Adjust benign/malicious traffic ratio
- Comprehensive accuracy analysis
- Per-class performance breakdown
- Statistical comparison between models

### 3. **Live Monitoring Simulation**
- Real-time network traffic simulation
- Continuous model performance monitoring
- Live accuracy and confidence plots
- Running statistics display
- Start/stop/clear monitoring controls

### 4. **Model Performance Comparison**
- Detailed model architecture comparison
- Training metrics and statistics
- Performance improvements analysis
- Parameter efficiency comparison

## 🔧 Technical Details

### Network Data Simulation
- Generates realistic CIC-IDS2017-style network features
- 15 attack types including DDoS, PortScan, Bot, Web Attacks, etc.
- Configurable anomaly levels for attack intensity
- 42 network features matching trained model input

### Model Integration
- Automatic loading of trained PyTorch models
- Support for both CPU and GPU inference
- Real-time prediction with confidence scores
- Batch processing capabilities

### Visualization
- Real-time matplotlib plots embedded in GUI
- Interactive controls and parameter adjustment
- Scrollable text displays for detailed results
- Tabbed interface for organized functionality

## 📊 Sample Use Cases

### Research & Development
- Test model robustness on different attack scenarios
- Compare model confidence on edge cases
- Analyze per-class performance differences
- Validate model behavior on simulated data

### Demonstration & Presentation
- Live demonstration of model capabilities
- Visual comparison of model performance
- Interactive exploration of detection accuracy
- Real-time monitoring simulation

### Educational
- Understand network intrusion detection concepts
- Explore machine learning model behavior
- Compare different architectural approaches
- Hands-on experience with cybersecurity ML

## 🛠️ Requirements

### Python Packages
```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tkinter (usually included with Python)
```

### Model Files (Auto-detected)
- `results/baseline/best_baseline_model.pth`
- `results/scs_id/scs_id_best_model.pth`

### Data Files (Optional)
- Model result files for detailed comparison
- Feature names and configuration data

## 🎮 Usage Examples

### Single Sample Test
1. Select attack type (e.g., "DDoS")
2. Adjust anomaly level (0.1 - 1.0)
3. Click "Generate Sample"
4. Click "Test Both Models"
5. Review predictions and confidence

### Batch Analysis
1. Set batch size (e.g., 100 samples)
2. Set benign ratio (e.g., 0.7 for 70% benign traffic)
3. Click "Run Batch Test"
4. Analyze overall and per-class accuracy

### Live Monitoring
1. Click "Start Monitoring"
2. Watch real-time accuracy plots
3. Monitor confidence trends
4. View running statistics

## 🔍 GUI Components

### Main Window Layout
- **Tabbed Interface**: Organized functionality
- **Control Panels**: Parameter adjustment
- **Result Displays**: Scrollable text areas
- **Live Plots**: Real-time visualization
- **Status Indicators**: Model loading status

### Interactive Elements
- Dropdown menus for attack type selection
- Sliders for parameter adjustment
- Buttons for actions and controls
- Text areas for detailed output
- Embedded matplotlib plots

## 📈 Performance Metrics

The GUI displays various performance metrics:
- **Accuracy**: Correct predictions / Total predictions
- **Confidence**: Model certainty in predictions
- **Per-class Performance**: Accuracy by attack type
- **Real-time Statistics**: Running averages and trends

## 🛡️ Security Note

This GUI generates simulated network data for testing purposes only. It does not process real network traffic or perform actual intrusion detection on live systems.

## 🤝 Contributing

To extend the GUI:
1. Add new attack types in `NetworkDataSimulator`
2. Implement additional metrics in `ModelTester`
3. Create new visualization tabs
4. Add export functionality for results

## 📞 Support

For issues or questions:
- Check model file paths and availability
- Verify Python package installations
- Review console output for error messages
- Ensure sufficient system resources for GUI rendering

---

**Built for**: Network Security Research & Education  
**Compatible with**: CIC-IDS2017 dataset models  
**Framework**: PyTorch + Tkinter + Matplotlib