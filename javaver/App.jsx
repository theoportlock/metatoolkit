// App.jsx (React Frontend)
import React, { useState } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import axios from 'axios';

const App = () => {
    const [elements, setElements] = useState([]);
    const [command, setCommand] = useState('');
    const [selectedNode, setSelectedNode] = useState(null);

    const addNode = () => {
        const id = `node-${elements.length}`;
        const newNode = { data: { id, label: command } };
        setElements([...elements, newNode]);
        setCommand('');
    };

    const removeSelected = () => {
        setElements(elements.filter((el) => el.data.id !== selectedNode));
        setSelectedNode(null);
    };

    const connectNodes = (source, target) => {
        const edge = { data: { source, target, id: `${source}-${target}` } };
        setElements([...elements, edge]);
    };

    const handleUpload = async (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await axios.post('/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            console.log('File uploaded:', response.data);
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    };

    const runCommand = async () => {
        if (!selectedNode) return alert('No node selected!');
        const node = elements.find((el) => el.data.id === selectedNode);
        try {
            const response = await axios.post('/execute', { command: node.data.label });
            alert('Command executed:\n' + response.data.output);
        } catch (error) {
            alert('Error executing command:\n' + error.response.data.error);
        }
    };

    return (
        <div>
            <div>
                <input
                    type="text"
                    value={command}
                    onChange={(e) => setCommand(e.target.value)}
                    placeholder="Enter bash command"
                />
                <button onClick={addNode}>Add Node</button>
                <button onClick={removeSelected}>Remove Selected</button>
                <button onClick={() => connectNodes('node-0', 'node-1')}>Connect Nodes</button>
                <input type="file" onChange={handleUpload} />
                <button onClick={runCommand}>Run Command</button>
            </div>
            <CytoscapeComponent
                elements={elements}
                style={{ width: '600px', height: '400px' }}
                cy={(cy) => {
                    cy.on('tap', 'node', (evt) => setSelectedNode(evt.target.id()));
                }}
                layout={{ name: 'grid' }}
            />
        </div>
    );
};

export default App;

