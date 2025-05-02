
// src/components/Navbar.js

import React from "react";
import "./Navbar.css";
import logo from "../assets/sheildLogo.png"; // adjust path based on your project

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo-container">
        <img src={logo} alt="SHEild Logo" className="navbar-logo" />
      </div>
      <div className="navbar-links">
        <a href="#home">Home</a>
        {/* Add more links if needed */}
      </div>
    </nav>
  );
};

export default Navbar;

