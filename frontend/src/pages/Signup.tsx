import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Header from "../components/Header";
import { useAuthStore } from "../store/auth";
import { signup as apiSignup } from "../api/client";
import "../styles/auth.css";

export default function Signup() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();
    const { login } = useAuthStore();

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        setError(null);
        try {
            const token = await apiSignup(email, password);
            login(token);
            navigate("/review");
        } catch (e: any) {
            setError(e?.response?.data?.detail ?? "Signup failed");
        }
    }

    return (
        <div className="auth-layout">
            <Header />
            <form className="auth-card" onSubmit={handleSubmit}>
                <h1>Sign Up</h1>
                {error && <div className="error">{error}</div>}
                <label>Email<input value={email} onChange={(e) => setEmail(e.target.value)} /></label>
                <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
                <button type="submit" className="primary">Create account</button>
                <p>Already have an account? <Link to="/login">Log in</Link></p>
            </form>
        </div>
    );
}


