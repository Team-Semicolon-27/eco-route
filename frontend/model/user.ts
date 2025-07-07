import {model, Model, models, Schema} from "mongoose";

export interface IUser extends Document {
    firstName: string;
    lastName: string;
    email: string;
    password: string;
    profile: string;
    role: string;
}

const UserSchema: Schema<IUser> = new Schema({
    firstName: { type: String, required: true },
    lastName: { type: String, required: true },
    email: { type: String, required: true },
    password: { type: String, required: true },
    profile: { type: String, required: true },
    role: { type: String, enum: ["driver", "manager"], required: true },
})

export const User: Model<IUser> = models.User || model<IUser>("User", UserSchema);

