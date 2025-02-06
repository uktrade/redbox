import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Body from "./Body"; // Your main component
import Chat from "./Chat"; // New component for /chat route

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Body />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
    </Router>
  );
}

export default App;
