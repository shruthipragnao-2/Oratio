import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Header from "../components/Header";
import { useAuthStore } from "../store/auth";
import { login as apiLogin } from "../api/client";
import "../styles/auth.css";

export default function Login() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();
    const { login } = useAuthStore();

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault();
        setError(null);
        try {
            const token = await apiLogin(email, password);
            login(token);
            navigate("/review");
        } catch (e: any) {
            setError(e?.response?.data?.detail ?? "Login failed");
        }
    }

    return (
        <div className="auth-layout">
            <Header />
            <form className="auth-card" onSubmit={handleSubmit}>
                <h1>Login</h1>
                {error && <div className="error">{error}</div>}
                <label>Email<input value={email} onChange={(e) => setEmail(e.target.value)} /></label>
                <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
                <button type="submit" className="primary">Login</button>
                <p>Don't have an account? <Link to="/signup">Sign up</Link></p>
            </form>
        </div>
    );
}


