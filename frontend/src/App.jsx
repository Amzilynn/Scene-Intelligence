import React, { useState, useEffect } from 'react';
import {
    Wifi,
    Users,
    TrendingUp,
    Clock,
    LayoutDashboard,
    Activity,
    Timer,
    MessageSquare,
    Hourglass,
    Zap,
    Map as MapIcon,
    BarChart3,
    Eraser
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';

import './App.css';

// Mock Data
const MOCK_TIMELINE = [
    { time: '10:00', interactions: 4 },
    { time: '11:00', interactions: 7 },
    { time: '12:00', interactions: 5 },
    { time: '13:00', interactions: 10 },
    { time: '14:00', interactions: 8 },
    { time: '15:00', interactions: 12 },
];

const MOCK_LOGS = [
    { id: 1, time: '23:59:13', persons: ['P-02', 'P-01'], type: 'Helping/Service' },
    { id: 2, time: '23:59:11', persons: ['P-03', 'P-05'], type: 'Helping/Service' },
];

function App() {
    const [activeTags, setActiveTags] = useState(['P-01', 'P-02', 'P-03', 'P-04']);
    const [timeframe, setTimeframe] = useState('15 min');

    return (
        <div className="dashboard-container">
            {/* Sidebar */}
            <aside className="sidebar">
                <header className="sidebar-header">
                    <div className="logo-container">
                        <LayoutDashboard className="logo-icon" />
                        <h1>SOCIAL INTERACTION ANALYSIS</h1>
                    </div>
                    <p className="mock-mode">
                        MOCK DATA MODE <span className="status-live">LIVE</span>
                    </p>
                </header>

                <section className="connectivity">
                    <div className="status-card">
                        <Wifi size={18} className="text-emerald-500" />
                        <span className="status-text">Connected</span>
                        <span className="status-dot pulse"></span>
                    </div>
                </section>

                <nav className="filters">
                    <div className="filter-group">
                        <h3><Users size={14} className="icon" /> PERSONS</h3>
                        <div className="person-tags">
                            {['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07', 'P-08'].map(id => (
                                <button
                                    key={id}
                                    className={`tag ${activeTags.includes(id) ? 'active' : ''}`}
                                    onClick={() => setActiveTags(prev =>
                                        prev.includes(id) ? prev.filter(t => t !== id) : [...prev, id]
                                    )}
                                >
                                    {id}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="filter-group">
                        <h3><Activity size={14} className="icon" /> INTERACTION TYPE</h3>
                        {['Approaching', 'Talking', 'Helping/Service', 'Waiting'].map(type => (
                            <label key={type} className="checkbox-container">
                                <input type="checkbox" defaultChecked={type === 'Approaching' || type === 'Talking'} />
                                <span className="checkmark"></span> {type}
                            </label>
                        ))}
                    </div>

                    <div className="filter-group">
                        <h3><Clock size={14} className="icon" /> TIMEFRAME</h3>
                        <div className="radio-group">
                            {['5 min', '15 min', '1 hour'].map(t => (
                                <button
                                    key={t}
                                    className={`btn-time ${timeframe === t ? 'active' : ''}`}
                                    onClick={() => setTimeframe(t)}
                                >
                                    {t}
                                </button>
                            ))}
                        </div>
                    </div>

                    <button className="btn-reset">
                        <Eraser size={16} className="icon" /> Reset Filters
                    </button>
                </nav>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                <div className="top-row">
                    <section className="live-feed card">
                        <header className="card-header">
                            <h2><Activity size={16} className="icon" /> LIVE FEED</h2>
                            <div className="feed-status">
                                <span className="badge-live"><Wifi size={12} className="inline mr-1" /> LIVE</span>
                            </div>
                        </header>
                        <div className="video-container">
                            <canvas id="feedCanvas"></canvas>
                            <div className="overlay-info">
                                <span className="badge-rec">REC</span>
                                <span className="track-count">4 tracked</span>
                            </div>
                        </div>
                    </section>

                    <aside className="interaction-log card">
                        <header className="card-header">
                            <h2><MessageSquare size={16} className="icon" /> INTERACTION LOG</h2>
                        </header>
                        <div className="log-table">
                            <div className="log-header">
                                <span>TIME</span>
                                <span>PERSONS</span>
                                <span>TYPE</span>
                            </div>
                            <div className="log-entries">
                                {MOCK_LOGS.map(log => (
                                    <div key={log.id} className="log-row">
                                        <span className="time">{log.time}</span>
                                        <span className="persons">{log.persons.join(', ')}</span>
                                        <span className="type-badge status-warning">{log.type}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </aside>
                </div>

                {/* KPI Row */}
                <section className="kpi-grid">
                    {[
                        { label: 'AVG APPROACH', value: '4.3', unit: 's', icon: Timer, color: 'text-cyan-400' },
                        { label: 'AVG DURATION', value: '18.2', unit: 's', icon: MessageSquare, color: 'text-emerald-400' },
                        { label: 'AVG WAIT', value: '20.4', unit: 's', icon: Hourglass, color: 'text-red-400' },
                        { label: 'SERVICE FREQ', value: '9', unit: '/hr', icon: Zap, color: 'text-amber-400' },
                    ].map((kpi, idx) => (
                        <div key={idx} className="kpi-card card">
                            <div className="kpi-info">
                                <span className="kpi-label">{kpi.label}</span>
                                <span className="kpi-value">{kpi.value} <small>{kpi.unit}</small></span>
                            </div>
                            <kpi.icon className={`kpi-icon ${kpi.color}`} size={24} />
                        </div>
                    ))}
                </section>

                {/* Analytics Row */}
                <div className="bottom-row">
                    <section className="analytics-timeline card">
                        <header className="card-header">
                            <h2><BarChart3 size={16} className="icon" /> TIMELINE</h2>
                        </header>
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={MOCK_TIMELINE}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2228" vertical={false} />
                                    <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                                    <YAxis stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0e1217', borderColor: '#1a2228', color: '#e2e8f0' }}
                                        itemStyle={{ color: '#06b6d4' }}
                                    />
                                    <Bar dataKey="interactions" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </section>

                    <section className="analytics-heatmap card">
                        <header className="card-header">
                            <h2><MapIcon size={16} className="icon" /> INTERACTION HEATMAP</h2>
                        </header>
                        <div className="heatmap-container">
                            {/* Heatmap Grid */}
                            <div className="grid grid-cols-10 gap-2 h-full p-2">
                                {Array.from({ length: 40 }).map((_, i) => (
                                    <div
                                        key={i}
                                        className="rounded-sm transition-all hover:scale-110"
                                        style={{
                                            backgroundColor: Math.random() > 0.7 ? (Math.random() > 0.5 ? '#f59e0b' : '#ef4444') : '#1a2228',
                                            opacity: 0.8
                                        }}
                                    ></div>
                                ))}
                            </div>
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
}

export default App;
