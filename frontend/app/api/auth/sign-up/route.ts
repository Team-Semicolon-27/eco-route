import {NextRequest, NextResponse} from "next/server";
import dbConnect from "@/lib/db";
import {User} from "@/model/user";
import bcrypt from "bcryptjs";
import {Manager} from "@/model/manager";
import mongoose from "mongoose";
import {ApiError} from "@/lib/apiError";
import {Driver} from "@/model/driver";

export async function POST(req: NextRequest) {
    await dbConnect();
    const mongoSession = await mongoose.startSession();
    mongoSession.startTransaction();
    try {
        const { firstName, lastName, email, password, profile, role, country, city, state } = await req.json();

        if (!firstName || !lastName || !email || !password || !profile || !role) {
            throw new ApiError("Data not found", 400);
        }

        if (role === "driver" && (!city || !state || !country)) {
            throw new ApiError("Address is required for driver", 400);
        }

        const existingUser = await User.findOne({email: email});

        if (existingUser) {
            throw new ApiError("User already exists", 409);
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const user = await User.create([{
            firstName,
            lastName,
            email,
            profile,
            password: hashedPassword,
            role
        }], {session: mongoSession})

        if (!user || user.length === 0) {
            throw new ApiError("failed to create user");
        }

        if (role === "manager") {
            const manager = await Manager.create([{
                userId: user[0]._id,
            }], {session: mongoSession});

            if (!manager || manager.length === 0) {
                throw new ApiError("failed to create manager");
            }
        } else {
            const driver = await Driver.create([{
                userId: user[0]._id,
                country,
                state,
                city
            }], {session: mongoSession});

            if (!driver || driver.length === 0) {
                throw new ApiError("failed to create driver");
            }
        }

        await mongoSession.commitTransaction();
        return NextResponse.json({ status: 200 });
    } catch (e) {
        await mongoSession.abortTransaction();
        return NextResponse.json(
            { error: e },
            { status: 500 }
        )
    } finally {
        await mongoSession.endSession();
    }
}