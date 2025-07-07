"use client";

import {useState, ChangeEvent, FormEvent, useEffect} from "react";
import {useRouter} from "next/navigation";
import Link from "next/link";
import axios, {AxiosError} from "axios";
import {City, Country, ICity, ICountry, IState, State} from "country-state-city";
import {CldImage, CldUploadButton, CloudinaryUploadWidgetResults} from "next-cloudinary";

interface SignupForm {
    firstName: string;
    lastName: string;
    email: string;
    password: string;
    role: "driver" | "manager" | "";
    country?: string;
    state?: string;
    city?: string;
}

export default function SignupPage() {
    const router = useRouter();

    const [form, setForm] = useState<SignupForm>({
        firstName: "",
        lastName: "",
        email: "",
        password: "",
        role: "",
        country: "",
        state: "",
        city: "",
    });

    const [profile, setProfile] = useState<string | null>(null);
    const [countries, setCountries] = useState<ICountry[]>([]);
    const [states, setStates] = useState<IState[]>([]);
    const [cities, setCities] = useState<ICity[]>([]);

    useEffect(() => {
        setCountries(Country.getAllCountries());
    }, []);

    useEffect(() => {
        if (form.country) {
            setStates(State.getStatesOfCountry(form.country));
            setForm((prev) => ({...prev, state: "", city: ""}));
        }
    }, [form.country]);

    useEffect(() => {
        if (form.country && form.state) {
            setCities(City.getCitiesOfState(form.country, form.state));
            setForm((prev) => ({...prev, city: ""}));
        }
    }, [form.state]);


    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const {name, value} = e.target;
        setForm((prev) => ({...prev, [name]: value}));
    };

    function handleUpload(result:  CloudinaryUploadWidgetResults) {
        if (result.event === "success" && result.info && typeof result.info !== "string") {
            setProfile(result.info.secure_url);
        }
    }

    const handleRoleChange = (role: "driver" | "manager") => {
        setForm((prev) => ({ ...prev, role: prev.role === role ? "" : role }));
    };

    const isEmailValid = (email: string): boolean => {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    };

    const isNameValid = (name: string): boolean => {
        return /^[a-zA-Z]{2,30}$/.test(name);
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setError("");
        setLoading(true);

        // 🛑 Validation
        if (!isNameValid(form.firstName)) {
            setError("First name must be 2-30 alphabetic characters.");
            setLoading(false);
            return;
        }

        if (!isNameValid(form.lastName)) {
            setError("Last name must be 2-30 alphabetic characters.");
            setLoading(false);
            return;
        }

        if (!isEmailValid(form.email)) {
            setError("Please enter a valid email address.");
            setLoading(false);
            return;
        }

        if (form.password.length < 6) {
            setError("Password must be at least 6 characters.");
            setLoading(false);
            return;
        }

        if (!form.role) {
            setError("Please select either Driver or Manager.");
            setLoading(false);
            return;
        }

        if (!profile) {
            setError("Please upload photo.");
            setLoading(false);
            return;
        }

        // ✅ Signup request
        try {
            const res = await axios.post("/api/auth/sign-up", {...form, profile});

            if (res.status === 200) {
                router.push("/sign-in");
            } else {
                setError(res.data.message || "Signup failed");
            }
        } catch (err) {
            const axiosErr = err as AxiosError<{ message: string }>;
            setError(axiosErr.response?.data?.message || "Signup failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-900 to-indigo-950 flex items-center justify-center px-4">
            <div className="bg-white/10 backdrop-blur-lg border border-white/20 p-8 rounded-2xl shadow-2xl w-full max-w-md">
                <h2 className="text-3xl font-bold text-white text-center mb-6">Create an Account</h2>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="flex space-x-2">
                        <input
                            name="firstName"
                            type="text"
                            placeholder="First Name"
                            value={form.firstName}
                            onChange={handleChange}
                            className="w-1/2 px-4 py-2 rounded-lg bg-white/20 text-white placeholder-gray-200 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                            required
                        />
                        <input
                            name="lastName"
                            type="text"
                            placeholder="Last Name"
                            value={form.lastName}
                            onChange={handleChange}
                            className="w-1/2 px-4 py-2 rounded-lg bg-white/20 text-white placeholder-gray-200 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                            required
                        />
                    </div>

                    <input
                        name="email"
                        type="email"
                        placeholder="Email"
                        value={form.email}
                        onChange={handleChange}
                        className="w-full px-4 py-2 rounded-lg bg-white/20 text-white placeholder-gray-200 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                        required
                    />

                    <input
                        name="password"
                        type="password"
                        placeholder="Password"
                        value={form.password}
                        onChange={handleChange}
                        className="w-full px-4 py-2 rounded-lg bg-white/20 text-white placeholder-gray-200 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                        required
                    />

                    {/* Upload Button */}
                    <CldUploadButton
                        uploadPreset={process.env.NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET!}
                        onSuccess={handleUpload}
                        className="bg-cyan-500 px-4 py-2 rounded-lg text-white"
                    >
                        Upload Profile Picture
                    </CldUploadButton>

                    {/* Preview */}
                    {profile && (
                        <CldImage
                            src={profile}
                            alt="Profile"
                            width={100}
                            height={100}
                            className="rounded-full mt-2"
                        />
                    )}


                    <div className="flex items-center justify-between text-white text-sm">
                        <label className="flex items-center space-x-2">
                            <input
                                type="checkbox"
                                checked={form.role === "driver"}
                                onChange={() => handleRoleChange("driver")}
                                className="accent-cyan-500"
                            />
                            <span>Driver</span>
                        </label>
                        <label className="flex items-center space-x-2">
                            <input
                                type="checkbox"
                                checked={form.role === "manager"}
                                onChange={() => handleRoleChange("manager")}
                                className="accent-cyan-500"
                            />
                            <span>Manager</span>
                        </label>
                    </div>

                    {form.role === "driver" && (
                        <>
                            <select name="country" value={form.country} onChange={handleChange} className="w-full px-4 py-2 rounded-lg bg-white/20 text-white" required>
                                <option value="">Select Country</option>
                                {countries.map((c) => (
                                    <option key={c.isoCode} value={c.isoCode}>
                                        {c.name}
                                    </option>
                                ))}
                            </select>

                            {states.length > 0 && (
                                <select name="state" value={form.state} onChange={handleChange} className="w-full px-4 py-2 rounded-lg bg-white/20 text-white" required>
                                    <option value="">Select State</option>
                                    {states.map((s) => (
                                        <option key={s.isoCode} value={s.isoCode}>
                                            {s.name}
                                        </option>
                                    ))}
                                </select>
                            )}

                            {cities.length > 0 && (
                                <select name="city" value={form.city} onChange={handleChange} className="w-full px-4 py-2 rounded-lg bg-white/20 text-white" required>
                                    <option value="">Select City</option>
                                    {cities.map((city) => (
                                        <option key={city.name} value={city.name}>
                                            {city.name}
                                        </option>
                                    ))}
                                </select>
                            )}
                        </>
                    )}

                    {error && <p className="text-red-400 text-sm">{error}</p>}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-cyan-500 hover:bg-cyan-600 text-white py-2 rounded-lg font-medium transition-colors duration-200"
                    >
                        {loading ? "Creating..." : "Sign Up"}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <p className="text-sm text-white/70">
                        Already have an account?{" "}
                        <Link href="/signin" className="text-cyan-300 font-semibold hover:underline">
                            Sign In
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
